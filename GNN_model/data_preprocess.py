import torch
import dgl
import numpy as np
import pandas as pd
import json


def get_raw_graph(dct):
    # TODO
    u, v = None
    g = dgl.graph((u, v))
    return g


def get_node_feat(dct):
    # TODO
    pass


def get_edge_feat(dct):
    # TODO
    pass


def read_graph(filepath):
    dct = {}
    with open(filepath, 'r') as f:
        dct = json.load(f)

    g = get_raw_graph(dct) # 读取图

    node_feat = get_node_feat(dct)
    g.ndata['name'] = node_feat['name']
    g.ndata['type'] = node_feat['type']
    g.ndata['label'] = node_feat['label']

    edge_data = get_edge_feat(dct)
    g.edata['score'] = edge_data['score']
    g.edata['relation'] = edge_data['relation']
    g.edata['timestamp'] = edge_data['timestamp']
    
    return g


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
    zeros = torch.zeros((int(0.9 * g.num_nodes()), 1))
    ones = torch.ones((int(0.1 * g.num_nodes()), 1))
    labels = torch.concat((ones, zeros), dim=0)
    labels = labels[torch.randperm(labels.size(0))]
    g.ndata['label'] = labels

    # 边有得分和关系两种特征，分别设为1维和50维
    g.edata['score'] = torch.randn(g.num_edges(), 1).type(torch.float32)
    g.edata['relation'] = torch.randn(g.num_edges(), 50).type(torch.float32)
    g.edata['timestamp'] = torch.randn(g.num_edges(), 1).type(torch.float32)

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
