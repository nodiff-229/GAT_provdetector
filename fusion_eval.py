from main import *


def get_true_cnt(positive_nodes, node):
    c = 0
    for pn in positive_nodes:
        if node == pn:
            c += 1
    return c

def evaluation_easy(graph_path_dir, predict_path):
    positive_nodes, negative_nodes = [], []

    for filename in os.listdir(graph_path_dir):
        graph_path = os.path.join(graph_path_dir, filename)
        with open(graph_path, "r") as f:
            g = json.load(f)
            for node in g['nodes']:
                if node['malicious_label'] == 'yes':
                    positive_nodes.append(reduce_len(node['id']))
                else:
                    negative_nodes.append(reduce_len(node['id']))

    with open(predict_path, "r") as ff:
        predict_dict = {}
        for r in map(lambda s: s.replace("\n", ""), ff.readlines()):
            if len(r) == 0:
                continue
            if r not in predict_dict:
                predict_dict[r] = 1
            else:
                predict_dict[r] += 1

    for positive_node in positive_nodes:
        if positive_node not in predict_dict:
            print(positive_node)


def evaluate(predict_path, graph_path):
    positive_nodes, negative_nodes = set(), set()

    with open(graph_path, "r") as f:
        g = json.load(f)
        for node in g['nodes']:
            if node['type'] != 1:
                continue
            if node['malicious_label'] == 'yes':
                positive_nodes.add(reduce_len(node['id']))
            else:
                negative_nodes.add(reduce_len(node['id']))

    with open(predict_path, "r") as ff:
        predict_set = set(map(lambda s: s.replace("\n", ""), ff.readlines()))

    tp, tn, fp, fn = 0, 0, 0, 0
    for pn in positive_nodes:
        if pn in predict_set:
            tp += 1
        else:
            fn += 1

    for ppn in predict_set:
        if len(ppn) == 0:
            continue
        if ppn not in positive_nodes:
            fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "p_cnt": len(positive_nodes)}

def eval_all(name, li):
    for item in li:
        print(f'result_{name}_{item}: '
              f'{evaluate(f"./result/result_{name}_{item}.txt", f"./data/fusion/compress-no-merge/{item}-compress-no-merge.json")}')


if __name__ == '__main__':
    k_5_list = ['M3', 'M4', 'M6', 'S1', 'S2', 'S3', 'S4']
    k_20_list = ['M2', 'M3', 'M4', 'M6', 'S1', 'S3', 'S4']
    k_215_list = ['M1', 'M2', 'M3', 'M6', 'S1', 'S2', 'S3']
    k_293_list = ['M1', 'M6', 'S1', 'S3']

    eval_all("k_5", k_5_list)
    eval_all("k_20", k_20_list)
    eval_all("k_215", k_215_list)
    eval_all("k_293", k_293_list)
    #
    # print(f'result_only_score: {evaluation("./data/fusion/compress-no-merge", "./result/result_only_score.txt")}')
