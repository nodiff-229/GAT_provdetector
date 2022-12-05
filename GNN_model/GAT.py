import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
import time
import numpy as np
import dgl
from data_preprocess import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载数据集
graph = fake_dataset()
graph = dgl.remove_self_loop(graph)
graph = dgl.add_self_loop(graph)
# 把graph搬到device上
graph = graph.to(device)

train_mask = graph.ndata['train_mask']
val_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
label = graph.ndata['label']

# features = graph.ndata['feat']

# 记录各个特征初始维度
all_dims = {'nfeat_name':graph.ndata['name'].shape[1],
           'nfeat_type':graph.ndata['type'].shape[1],
           'efeat_relation':graph.edata['relation'].shape[1],
           'efeat_score':graph.edata['score'].shape[1],
           'efeat_timestamp':graph.edata['timestamp'].shape[1],}

# 隐藏层维度
n_hidden = 100

# 输出维度
n_classes = 1

# 注意力头数
num_heads = 8

# dropout系数
feat_drop = 0.6

# attention dropout系数
attn_drop = 0.5

lr = 0.01
weight_deacy = 3e-4
num_epochs = 500

# 将节点的两个特征concate到一起
input_features = torch.concat((graph.ndata['name'],graph.ndata['type']), dim=-1)
in_dim = input_features.shape[1]

# 真正的GAT操作
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # 公式 (1)
        self.fc_nfeat = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_efeat_relation = nn.Linear(all_dims['efeat_relation'], out_dim, bias=False)
        self.fc_efeat_score = nn.Linear(all_dims['efeat_score'], out_dim, bias=False)
        self.fc_efeat_timestamp = nn.Linear(all_dims['efeat_timestamp'], out_dim, bias=False)

        # 公式 (2)
        self.attn_fc = nn.Linear(5 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['e_feat_all']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    # 消息函数
    def message_func(self, edges):
        # 公式 (3), (4)所需，传递消息用的用户定义函数
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

        # 论文公式 (1)
        z = self.fc_nfeat(h)
        transformed_efeat_relation = self.fc_efeat_relation(self.g.edata['relation'])
        transformed_efeat_score = self.fc_efeat_score(self.g.edata['score'])
        transformed_efeat_timestamp = self.fc_efeat_timestamp(self.g.edata['timestamp'])

        e_feat_all = torch.concat((transformed_efeat_relation,transformed_efeat_score,transformed_efeat_timestamp),dim=-1)

        self.g.ndata['z'] = z
        self.g.edata['e_feat_all'] = e_feat_all
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

    def forward(self, graph,h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


def evaluate(model, graph,features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph,features)
        logits = logits[mask]
        labels = labels[mask]
        # _, indices = torch.max(logits, dim=1)
        logits = F.sigmoid(logits)
        predicts = torch.where(logits > 0.5, 1, 0)
        correct = torch.sum(predicts == labels)
        return correct.item() * 1.0 / len(labels)




# 加载数据集
# dataset = CoraGraphDataset('../cora')
# graph = dataset[0]
# graph = dgl.remove_self_loop(graph)
# graph = dgl.add_self_loop(graph)
# 把graph搬到gpu上
# graph = graph.to(device)

# train_mask = graph.ndata['train_mask']
# val_mask = graph.ndata['val_mask']
# test_mask = graph.ndata['test_mask']
# label = graph.ndata['label']
# features = graph.ndata['feat']




model = GAT(graph,
            in_dim=in_dim,
            hidden_dim=n_hidden,
            out_dim=n_classes,
            num_heads=num_heads)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_deacy)
criterion = torch.nn.BCEWithLogitsLoss()
dur = []

for epoch in range(num_epochs):
    if epoch >= 3:
        t0 = time.time()

    logits = model(graph,input_features)
    # logp = F.log_softmax(logits, 1)
    # loss = F.nll_loss(logp[train_mask], label[train_mask])
    # loss = criterion(logits[train_mask].unsqueeze(1), label[train_mask].unsqueeze(1).type(torch.float))
    loss = criterion(logits[train_mask], label[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc_val = evaluate(model, graph,input_features, label, val_mask)

    x = epoch
    y_loss = loss.cpu().detach().numpy()
    y_acc = acc_val

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | Accuracy {:.4f}".format(
        epoch, loss.item(), np.mean(dur), acc_val))

acc_test = evaluate(model, graph,input_features, label, test_mask)
print("Test Accuracy {:.4f}".format(acc_test))
