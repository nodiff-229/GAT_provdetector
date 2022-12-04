from main import *

# 1:pid
# 2:ip
# 3:obj
# 5:query
# 6:host_domain
# 8:file
NODE_TYPES = {
    0: "user", 1: "process", 2: "socket", 3: "obj", 5: "query", 6: "host_domain", 8: "file"
}

def save_json(filepath, obj):
    with open(filepath, 'w') as f:
        json.dump(obj, f)

def load_on_json(filepath: str):
    with open(filepath, "r") as f:
        g = json.load(f)
    MDG = nx.MultiDiGraph()
    node_dict = {}
    for node in g['nodes']:
        node_dict[reduce_len(node['id'])] = {
            'node_type': NODE_TYPES[node["type"]],
            'name': reduce_len(node['name']),
            'malicious': not node['malicious_label'] == 'no'
        }
    for link in g["links"]:
        name_start, name_end, relation_type, time, success = reduce_len(link["source"]), reduce_len(link["target"]), \
            link["rel"], link["ts"], True
        # if isIpv4(name_start) or isIpv4(name_end):
        #     continue
        time = int(time * 1000)
        if name_start not in MDG.nodes:
            MDG.add_node(name_start, type=node_dict[name_start]['node_type'] if name_start in node_dict else "")
        if name_end not in MDG.nodes:
            MDG.add_node(name_end, type=node_dict[name_end]['node_type'] if name_end in node_dict else "")
        MDG.add_edge(name_start, name_end, relation_type=relation_type, time=time, success=success)

    # 获取良性和恶性节点名列表
    positive_nodes = []
    negative_nodes = []
    for node in g['nodes']:
        if node['malicious_label'] == 'yes':
            positive_nodes.append(reduce_len(node['id']))
        else:
            negative_nodes.append(reduce_len(node['id']))

    return MDG, positive_nodes, negative_nodes


def start(graph_path):
    timer = time.time()
    sql_engine = create_engine(db_config['uri'])

    # 0. 历史数据入库
    # fusion_data.py

    # 1. 打分
    MDG, positive_nodes, negative_nodes = load_on_json(graph_path)
    markRegularScore(MDG, sql_engine)
    timer = timer_check("Score", timer)

    # nx.write_gpickle(MDG, "./temp/fusion_scored_MDG.pkl")
    # MDG: nx.MultiDiGraph = nx.read_gpickle("./bugfix/temp/fusion_scored_MDG.pkl")

    # 2. 转换DAG
    add_source_and_sink(MDG)
    convertToDAG(MDG)
    timer = timer_check("Convert to DAG", timer)

    # 3. 选择K Longest Path
    DG = multi2Single(MDG)

    return DG, { 'pos': positive_nodes, 'neg': negative_nodes }


def evaluation(graph_path, predict):
    with open(graph_path, "r") as f:
        g = json.load(f)
    positive_nodes, negative_nodes = [], []

    for node in g['nodes']:
        if node['malicious_label'] == 'yes':
            positive_nodes.append(reduce_len(node['id']))
        else:
            negative_nodes.append(reduce_len(node['id']))

    outs = predict['outliner'] + predict['inliner']
    out_set = set()
    tp, tn, fp, fn = 0, 0, 0, 0
    for p_node in positive_nodes:
        flag = False
        for out in outs:
            if p_node in out:
                out_set.add(out)
                flag = True
        if flag:
            tp += 1
        else:
            fn += 1

    fp = len(outs) - len(out_set)
    tn = len(negative_nodes) - fp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == '__main__':
    out_dir = './out/compress-hw'
    hosts = ['S1', 'S2', 'S3', 'S4', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    eval_ress = {}
    for host in hosts:
        print(f"******** Starting {host} ********")
        p, nodes_label = start(f'./data/tokenized/compressed_pp/{host}-compress.json')
        nx.write_gpickle(p, f"./GNN_data/{host}_Scored.pkl")
        save_json(f"./GNN_data/{host}_NodeLabel.json", nodes_label)
