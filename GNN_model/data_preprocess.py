import torch
import dgl
import numpy as np
import pandas as pd


def fake_dataset():
    """
    构造假图
    :return: graph
    """

    u = torch.randint(0, 600, (1500,))
    v = torch.randint(0, 600, (1500,))

    g = dgl.graph((u, v))
    # 节点有名字和类型两种特征，均设为50维
    g.ndata['name'] = torch.randn(g.num_nodes(), 50).type(torch.float32)
    g.ndata['type'] = torch.randn(g.num_nodes(), 50).type(torch.float32)


    # 假标签，假设含有10%的异常节点
    ones = torch.ones((int(0.9 * g.num_nodes()), 1))
    zeros = torch.zeros((int(0.1 * g.num_nodes()), 1))
    labels = torch.concat((ones, zeros), dim=0)
    labels = labels[torch.randperm(labels.size(0))]
    g.ndata['label'] = labels

    # 边有得分和关系两种特征，分别设为1维和50维
    g.edata['score'] = torch.randn(g.num_edges(), 1).type(torch.float32)
    g.edata['relation'] = torch.randn(g.num_edges(), 50).type(torch.float32)

    # 划分训练集，验证集和测试集
    # 训练集60%，验证集10%，测试集30%
    train_mask = torch.zeros((g.num_nodes()))
    train_mask[:int(0.6 * g.num_nodes())] = True
    g.ndata['train_mask'] = train_mask.bool()

    val_mask = torch.zeros((g.num_nodes()))
    val_mask[int(0.6 * g.num_nodes()):int(0.7 * g.num_nodes())] = True
    g.ndata['val_mask'] = val_mask.bool()

    test_mask = torch.zeros((g.num_nodes()))
    test_mask[int(0.7 * g.num_nodes()):] = True
    g.ndata['test_mask'] = test_mask.bool()
    return g


if __name__ == '__main__':
    g = fake_dataset()

