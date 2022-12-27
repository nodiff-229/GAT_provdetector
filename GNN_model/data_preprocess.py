import torch
import dgl
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from transformers import logging
logging.set_verbosity_error()


def get_raw_graph(dct):
    # 先构建节点名字到idx的映射
    name_to_idx = defaultdict(int)
    idx = 0
    for node in dct['nodes']:
        assert not node['id'] in name_to_idx.keys()  # 避免有节点名字重复
        name_to_idx[node['id']] = idx
        idx += 1

    # 构建边
    edge_num = len(dct['edges'])
    u = torch.zeros((edge_num,)).type(torch.int)
    v = torch.zeros((edge_num,)).type(torch.int)
    for i in range(edge_num):
        edge = dct['edges'][i]
        assert edge['src'][0] in name_to_idx.keys()
        assert edge['target'][0] in name_to_idx.keys()
        src_idx = name_to_idx[edge['src'][0]]
        target_idx = name_to_idx[edge['target'][0]]
        u[i] = src_idx
        v[i] = target_idx

    g = dgl.graph((u, v))
    return g


def get_node_feat(dct, embed_len=50):
    nodes = dct['nodes']
    nodes_num = len(nodes)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()

    name = torch.zeros((nodes_num, embed_len)).type(torch.float32)
    type = torch.zeros((nodes_num, embed_len)).type(torch.float32)
    label = torch.zeros((nodes_num, 1)).type(torch.float32)

    for i, node in enumerate(nodes):
        name_token = tokenizer.tokenize(node['id'])
        type_token = tokenizer.tokenize("[CLS] " + " [SEP] ".join(node['anonymous']) + " [SEP]")

        indexed_name_tokens = tokenizer.convert_tokens_to_ids(name_token)
        name_segments_ids = [1] * len(name_token)
        indexed_type_tokens = tokenizer.convert_tokens_to_ids(type_token)
        type_segments_ids = [1] * len(type_token)

        name_token_tensor = torch.tensor(indexed_name_tokens)
        type_token_tensor = torch.tensor(indexed_type_tokens)

        with torch.no_grad():

            name_outputs = model(name_token_tensor, name_segments_ids)
            name_hidden_states = name_outputs[2]

            type_outputs = model(type_token_tensor, type_segments_ids)
            type_hidden_states = type_outputs[2]

            name_embeddings = torch.stack(name_hidden_states, dim=0)
            name_embeddings = torch.squeeze(name_embeddings, dim=1)
            name_embeddings = name_embeddings.permute(1, 0, 2)
            name_embed = torch.zeros((name_embeddings.shape[0], name_embeddings.shape[-1]))

            for i in range(name_embeddings.shape[0]):
                token = name_embeddings[i]

                sum_vec = torch.sum(token[-4:], dim=0)
                name_embed[i] = sum_vec

            name_embed = name_embed.flatten()

            type_token_vecs = type_hidden_states[-2][0]
            type_embed = torch.mean(type_token_vecs, dim=0)

        name[i] = name_embed
        type[i] = type_embed
        label[i] = 0 if node['malicious'] == 'false' else 1

    return {
        'name': name,
        'type': type,
        'label': label
    }


def get_node_feat1(dct, embed_len=768):
    nodes = dct['nodes']
    nodes_num = len(nodes)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()

    name = torch.zeros((nodes_num, embed_len)).type(torch.float32)
    type = torch.zeros((nodes_num, embed_len)).type(torch.float32)
    label = torch.zeros((nodes_num, 1)).type(torch.float32)

    for i in tqdm(range(len(nodes))):
        node = nodes[i]
        name_token = tokenizer.tokenize("[CLS] " + " [SEP] ".join(node['id'][:512].split('/')) + " [SEP]")
        type_token = tokenizer.tokenize("[CLS] " + " [SEP] ".join(node['anonymous']) + " [SEP]")

        indexed_name_tokens = tokenizer.convert_tokens_to_ids(name_token)
        name_segments_ids = [1] * len(name_token)
        indexed_type_tokens = tokenizer.convert_tokens_to_ids(type_token)
        type_segments_ids = [1] * len(type_token)

        name_token_tensor = torch.tensor(indexed_name_tokens).view(1, -1)
        type_token_tensor = torch.tensor(indexed_type_tokens).view(1, -1)
        name_segments_ids = torch.tensor(name_segments_ids).view(1, -1)
        type_segments_ids = torch.tensor(type_segments_ids).view(1, -1)

        with torch.no_grad():

            name_outputs = model(name_token_tensor, name_segments_ids)
            name_hidden_states = name_outputs[2]

            type_outputs = model(type_token_tensor, type_segments_ids)
            type_hidden_states = type_outputs[2]

            name_token_vecs = name_hidden_states[-2][0]
            name_embed = torch.mean(name_token_vecs, dim=0)
            type_token_vecs = type_hidden_states[-2][0]
            type_embed = torch.mean(type_token_vecs, dim=0)

        name[i] = name_embed
        type[i] = type_embed
        label[i] = 1 if node['malicious'] else 0
    return {
        'name': name,
        'type': type,
        'label': label
    }


def get_edge_feat(dct):
    # 统计只有这三种relation，所以用简单的3维one-hot表示
    relations = {
        'down': 0,
        'transport_remote_ip': 1,
        'refer': 2
    }
    edge_feat = {}
    edge_num = len(dct['edges'])
    timestamp = torch.zeros((edge_num, 1)).type(torch.float32)
    score = torch.zeros((edge_num, 1)).type(torch.float32)
    relation = torch.zeros((edge_num, 3)).type(torch.float32)
    for i in range(edge_num):
        edge = dct['edges'][i]
        timestamp[i] = edge['time']
        score[i] = edge['w_score']
        relation[i][relations[edge['rela']]] = 1
    edge_feat['timestamp'] = timestamp
    edge_feat['score'] = score
    edge_feat['relation'] = relation
    return edge_feat


def read_graph(filepath):
    dct = {}
    with open(filepath, 'r') as f:
        dct = json.load(f)

    g = get_raw_graph(dct)  # 读取图

    node_feat = get_node_feat1(dct)

    # 造假数据拉通用，注释掉
    # node_feat = {
    #     'name': torch.randn(g.num_nodes(), 50).type(torch.float32),
    #     'type': torch.randn(g.num_nodes(), 50).type(torch.float32),
    #     'label': torch.ones(g.num_nodes(), 1)
    # }

    g.ndata['name'] = node_feat['name']
    g.ndata['type'] = node_feat['type']
    g.ndata['label'] = node_feat['label']

    edge_data = get_edge_feat(dct)
    g.edata['score'] = edge_data['score']
    g.edata['relation'] = edge_data['relation']
    g.edata['timestamp'] = edge_data['timestamp']

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
    # 训练集30%，验证集100%，测试集100%
    train_mask = torch.zeros((g.num_nodes()))
    train_mask[:int(0.7 * g.num_nodes())] = True
    g.ndata['train_mask'] = train_mask.bool()

    val_mask = torch.zeros((g.num_nodes()))
    val_mask[:int(1 * g.num_nodes())] = True
    g.ndata['val_mask'] = val_mask.bool()

    test_mask = torch.zeros((g.num_nodes()))
    test_mask[:int(1 * g.num_nodes())] = True
    g.ndata['test_mask'] = test_mask.bool()
    return g


if __name__ == '__main__':
    # g = fake_dataset()
    hosts = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'S1', 'S2', 'S3', 'S4']
    # 保存图的二进制文件
    for host in hosts:
        print(f'processing {host}')
        g = read_graph(f'../GNN_data_integrate/{host}.json')
        dgl.save_graphs(f'../GNN_graph/{host}.bin', g)
    # 读取图二进制文件
    for host in hosts:
        g, _ = dgl.load_graphs(f'../GNN_graph/{host}.bin')
        print(g[0].num_nodes())


