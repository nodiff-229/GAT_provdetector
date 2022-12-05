import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
import time
import numpy as np
import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 真正的GAT操作
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # 公式 (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # TODO 这里计算attention系数要concate边特征
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    # 消息函数
    def message_func(self, edges):
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        # TODO 这里传递消息时要把边特征同时传递，我记得应该是在之前的处理中把边特征concate到z里，待确认
        return {'z': edges.src['z'], 'e': edges.data['e']}

    # 聚合函数
    def reduce_func(self, nodes):
        # 论文公式 (3), (4)所需, 归约用的用户定义函数
        # 论文公式 (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 论文公式 (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # TODO 与点特征类似，加入边特征维度的转换
        # 论文公式 (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # 论文公式 (2)
        self.g.apply_edges(self.edge_attention)
        # 论文公式 (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


# 多头注意力层
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


# TODO 加载自建图，现在先造个假图

# 加载数据集
dataset = CoraGraphDataset('../cora')
graph = dataset[0]
graph = dgl.remove_self_loop(graph)
graph = dgl.add_self_loop(graph)
# 把graph搬到gpu上
graph = graph.to(device)

train_mask = graph.ndata['train_mask']
val_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
label = graph.ndata['label']
features = graph.ndata['feat']

in_feats = features.shape[1]
n_hidden = 8
n_classes = dataset.num_classes
num_heads = 8
feat_drop = 0.6
attn_drop = 0.5
lr = 0.02
weight_deacy = 3e-4
num_epochs = 50

model = GAT(graph,
            in_dim=features.shape[1],
            hidden_dim=n_hidden,
            out_dim=7,
            num_heads=num_heads)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_deacy)

dur = []

for epoch in range(num_epochs):
    if epoch >= 3:
        t0 = time.time()

    logits = model(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], label[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc_val = evaluate(model, features, label, val_mask)

    x = epoch
    y_loss = loss.cpu().detach().numpy()
    y_acc = acc_val

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | Accuracy {:.4f}".format(
        epoch, loss.item(), np.mean(dur), acc_val))

acc_test = evaluate(model, features, label, test_mask)
print("Test Accuracy {:.4f}".format(acc_test))
