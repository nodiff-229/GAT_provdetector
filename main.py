import json
import math
import re
import time
from datetime import datetime

import networkx as nx
import numpy as np
import pandas
import pangres
from sqlalchemy import create_engine

from commons.embedding import *
from commons.ep import find_eppstein_ksp
from commons.lof import *


def reduce_len(s):
    return s[:2000] if len(s) > 2000 else s


"""
    每个tsv文件对应的八种关系可以读入溯源图，转换成对应的点或边。
    对于八种关系的第1、5、6、7种，采用索引节点，即只存储文件名和对应的行号，需要获得数据时回到文件进行反查
    对于八种关系的第2、3、4、8种，采用正常节点，将信息直接存储在节点中
    当解析出一个节点，应当先判断其是否存在。
"""
with open("./db.json", 'r') as db_config_f:
    db_config = json.load(db_config_f)


def load_on_tsv(file_path: str, host='default', save_db=False, sql_engine=None):
    # 从一个 tsv文件中加载各类图节点
    file_basename = os.path.basename(file_path)
    dfs = pandas.read_csv(file_path, sep='\t', chunksize=100000)
    if save_db and not sql_engine:
        raise Exception("未传入sql_engine")

    # 复合有向图
    MDG = nx.MultiDiGraph()

    # 每批数据
    for df in dfs:
        # 每行数据
        df['host'] = host
        in_process = []
        out_process = []

        for row in df.itertuples():
            line = row.Index
            start, end, time, relation_type, success = row.start, row.end, row.time, row.relation_type, row.success

            # 节点类型
            from_type, to_type = row.from_type, row.to_type
            # 判断start和end是否为空
            start_nan, end_nan = pandas.isna(start), pandas.isna(end)
            relation_type = None if pandas.isna(relation_type) else str(relation_type)

            if not start_nan:
                # 起始节点名
                name_start = str(start)
            if not end_nan:
                # 终止节点名
                name_end = str(end)

            if not start_nan:
                MDG.add_node(name_start, type=from_type)
            if not end_nan:
                MDG.add_node(name_end, type=to_type)
            MDG.add_edge(name_start, name_end, relation_type=relation_type, time=time, success=success)

            if not (len(str(time)) == 10 or len(str(time)) == 13):
                raise Exception("时间戳长度不为10或13")
            date_str = datetime.fromtimestamp(time if len(str(time)) == 10 else time / 1000).strftime("%Y-%m-%d")
            # TODO: 分辨哪些是进程；只有进程才写入数据库
            in_process.append([name_end, date_str, host, str(relation_type)])
            out_process.append([name_start, date_str, host, str(relation_type)])

        if save_db:
            insert_sql(in_process, out_process, df, sql_engine)
    return MDG


"""
历史数据入库
"""


def insert_sql(in_process, out_process, df, sql_engine):
    in_process_df = pandas.DataFrame(in_process, columns=["entity", "date", "host", "relation_type"])
    out_process_df = pandas.DataFrame(out_process, columns=["entity", "date", "host", "relation_type"])

    in_process_df = in_process_df.drop_duplicates(["entity", "date", "host", "relation_type"])
    out_process_df = out_process_df.drop_duplicates(["entity", "date", "host", "relation_type"])

    in_process_df.reset_index(drop=True, inplace=True)
    out_process_df.reset_index(drop=True, inplace=True)

    in_process_df['dumb'] = ''
    out_process_df['dumb'] = ''

    in_process_df.set_index(['entity', 'date', 'host', "relation_type"], inplace=True, drop=True)
    out_process_df.set_index(['entity', 'date', 'host', "relation_type"], inplace=True, drop=True)

    df[["time", "host", "start", "end", "relation_type"]] \
        .to_sql('events', sql_engine, if_exists='append', index=False)
    pangres.upsert(sql_engine, in_process_df, "in_process", if_row_exists="ignore")
    pangres.upsert(sql_engine, out_process_df, "out_process", if_row_exists="ignore")


def add_source_and_sink(mdg: nx.MultiDiGraph):
    # 入度列表
    inDegrees = mdg.in_degree()
    # 出度列表
    outDegrees = mdg.out_degree()

    # 入度为0的点的列表
    node_0_indegree = []

    # 出度为0的点的列表
    node_0_outdegree = []

    for ind in inDegrees:
        node_name = ind[0]
        indegree = ind[1]
        if indegree == 0:
            node_0_indegree.append(node_name)

    for outd in outDegrees:
        node_name = outd[0]
        outdegree = outd[1]
        if outdegree == 0:
            node_0_outdegree.append(node_name)

    if 'V_source' in node_0_indegree:
        node_0_indegree.remove('V_source')

    if 'V_sink' in node_0_outdegree:
        node_0_outdegree.remove('V_sink')

    for node in node_0_indegree:
        mdg.add_edge("V_source", node, time=-1, relation_type='source edge', success='success', w_score=0)

    for node in node_0_outdegree:
        mdg.add_edge(node, "V_sink", time=float('inf'), relation_type='sink edge', success='success', w_score=0)

    return


"""
根据节点名称获得入边
"""


def getInEdges(mdg, node):
    in_edges = list(
        map(lambda x: (x[0], x[1], x[2], mdg[x[0]][x[1]][x[2]]), filter(lambda x: x[1] == node, mdg.out_edges)))

    # 给每条边添加一个入边标记
    for edge in in_edges:
        edge[3]["edge_direction"] = 0

    return in_edges


"""
根据节点名称获得出边
"""


def getOutEdges(mdg, node):
    out_edges = list(
        map(lambda x: (x[0], x[1], x[2], mdg[x[0]][x[1]][x[2]]), filter(lambda x: x[0] == node, mdg.out_edges)))
    # 给每条边添加一个出边标记
    for edge in out_edges:
        edge[3]["edge_direction"] = 1
    return out_edges


def add_edge_direction(edge, direction):
    edge[2]["edge_direction"] = direction
    return edge


"""
将有向图转变为有向无环图
"""


def convertToDAG(mdg: nx.MultiDiGraph):
    # 记录是节点否被访问过
    visited = {}

    originNodes = list(mdg.nodes)

    for node in originNodes:
        if node == "V_source" or node == "V_sink":
            visited[node] = True
        else:
            visited[node] = False
    # index = 0
    for node in originNodes:
        if visited[node] == True:
            continue
        else:
            visited[node] = True
            # timer = time.time()
            # 入边
            # in_edges = getInEdges(mdg, node)
            in_edges = list(mdg.in_edges(node, data=True))
            in_edges = [add_edge_direction(edge, 0) for edge in in_edges]

            # for edge in in_edges:
            #     edge[2]["edge_direction"] = 0
            # 出边
            # out_edges = getOutEdges(mdg, node)
            out_edges = list(mdg.out_edges(node, data=True))
            out_edges = [add_edge_direction(edge, 1) for edge in out_edges]

            # for edge in out_edges:
            #     edge[2]["edge_direction"] = 1
            # 对所有边的时间进行排序
            all_deges = in_edges + out_edges

            all_deges = sorted(all_deges, key=lambda x: (x[2]['time'], -x[2]['edge_direction']))
            # timer = timer_check("出入边排序", timer)

            # timer = time.time()

            # 寻找分裂点上边的区间
            start = 0
            end = 0
            i = 0
            counter = 0  # 分裂出的点的后缀
            while i < len(all_deges):
                # 寻找由出边到入边的转折点
                if i + 1 < len(all_deges) and all_deges[i][0] == node and all_deges[i + 1][1] == node:
                    end = i + 1
                    interval = all_deges[start:end]
                    # 对点进行分裂，添加边
                    j = 0
                    split_node_name = node + "_" + str(counter)
                    mdg.add_node(split_node_name, type=mdg.nodes[node]['type'])
                    counter += 1
                    while j < len(interval):
                        # 增加入边
                        if interval[j][1] == node:
                            mdg.add_edge(interval[j][0], split_node_name, relation_type=interval[j][2]['relation_type'],
                                         time=interval[j][2]['time'], success=interval[j][2]['success'],
                                         w_score=interval[j][2]['w_score'])
                            # mdg.remove_edge(interval[j][0], node)

                        # 增加出边
                        elif interval[j][0] == node:
                            mdg.add_edge(split_node_name, interval[j][1], relation_type=interval[j][2]['relation_type'],
                                         time=interval[j][2]['time'], success=interval[j][2]['success'],
                                         w_score=interval[j][2]['w_score'])
                            # mdg.remove_edge(node, interval[j][1])

                        j += 1
                    # mdg.add_nodes_from(interval)

                    start = i + 1
                    i += 1
                    continue

                i += 1

            end = len(all_deges)
            interval = all_deges[start:end]
            # 对点进行分裂，添加边
            j = 0
            split_node_name = node + "_" + str(counter)
            mdg.add_node(split_node_name, type=mdg.nodes[node]['type'])
            counter += 1
            while j < len(interval):
                # 增加入边
                if interval[j][1] == node:
                    mdg.add_edge(interval[j][0], split_node_name, relation_type=interval[j][2]['relation_type'],
                                 time=interval[j][2]['time'], success=interval[j][2]['success'],
                                 w_score=interval[j][2]['w_score'])
                    # mdg.remove_edge(interval[j][0], node)
                elif interval[j][0] == node:
                    mdg.add_edge(split_node_name, interval[j][1], relation_type=interval[j][2]['relation_type'],
                                 time=interval[j][2]['time'], success=interval[j][2]['success'],
                                 w_score=interval[j][2]['w_score'])
                    # mdg.remove_edge(node, interval[j][1])

                j += 1

            # 删除原节点
            mdg.remove_node(node)
            # print(index, "\n")
            # index += 1
            # timer = timer_check("图分裂", timer)
            # print(1)

    # 分裂出来的点可能会有没有入边或者没有出边的情况，感觉这些点得和源点和汇点连接
    add_source_and_sink(mdg)


"""
计算Regular Score
"""


def replaceWildcard(s):
    return re.sub(r"\*+", "%", s)


def calcRegularScore(start, end, relation_type, timestamp, min_date, max_date, sql_engine, host_cnt):
    print(".", end='')
    with sql_engine.connect() as connection:
        if not (len(str(timestamp)) == 10 or len(str(timestamp)) == 13):
            raise Exception("时间戳长度不为10或13")
        curr_date = datetime.fromtimestamp(timestamp if len(str(timestamp)) == 10 else timestamp / 1000).date()
        if isinstance(max_date, str):
            max_date = datetime.strptime(max_date, "%Y-%m-%d")
        if isinstance(min_date, str):
            min_date = datetime.strptime(min_date, "%Y-%m-%d")

        start = replaceWildcard(start)
        end = replaceWildcard(end)

        tot_date_cnt = (max_date - min_date).days + 1
        res = connection.execute(
            """(select COUNT(DISTINCT host) from provdetector.public.events
               where start LIKE %s and "end" LIKE %s and relation_type = %s)
               UNION ALL
               (select COUNT(DISTINCT "date") from provdetector.public.in_process AS cnt
               where entity LIKE %s AND relation_type = %s)
               UNION ALL
               (select COUNT(DISTINCT "date") from provdetector.public.out_process AS cnt
               where entity LIKE %s AND relation_type = %s)""",
            start, end, relation_type,
            end, relation_type,
            start, relation_type)
        lRes = list(res)
        H_e = lRes[0][0]
        H = host_cnt

        """ M = H(e)/H """
        M = H_e / H

        """ In/Out """
        T_from = tot_date_cnt - lRes[1][0]
        T_to = tot_date_cnt - lRes[2][0]
        IN = T_to / tot_date_cnt
        OUT = T_from / tot_date_cnt
    rs = OUT * M * IN
    if rs == 0:
        return 1e-9
    else:
        return rs


def markRegularScore(mdg: nx.MultiDiGraph, sql_engine, weight=1):
    cnt = 0
    ct = time.time()
    with sql_engine.connect() as connection:
        res = connection.execute("""
        select min(bar.a) from (
            select min("date") as a from provdetector.public.in_process
            union
            select min("date") as a from provdetector.public.out_process
        ) as bar
        """)
        min_date = list(res)[0][0]
        res = connection.execute("""
        select max(bar.a) from (
            select max("date") as a from provdetector.public.in_process
            union
            select max("date") as a from provdetector.public.out_process
        ) as bar
        """)
        max_date = list(res)[0][0]
        res = connection.execute(
            """select count(distinct host) from provdetector.public.events""")
        host_cnt = list(res)[0][0]

    score_list = []
    for start, end, edge in mdg.edges(data=True):
        if cnt % 1000 == 0:
            print(f"{cnt} / {len(mdg.edges)}, in {time.time() - ct}s")
            ct = time.time()
        relation_type = edge['relation_type']
        timestamp = edge['time']
        rs = calcRegularScore(start, end, relation_type, timestamp,
                              min_date=min_date, max_date=max_date, sql_engine=sql_engine, host_cnt=host_cnt)
        # to -log, shortest PRODUCTION to longest SUM
        # TODO: log or -log??
        # ws = -math.log2(rs)

        # 测试用
        # ws = -1 * weight * math.log2(rs)
        ws = weight * math.log2(rs)
        score_list.append({"start": start, "end": end, "ws": ws})
        edge['w_score'] = ws
        cnt += 1
    score_list = sorted(score_list, key=lambda t: t["ws"])
    print()


def multi2Single(MDG: nx.MultiDiGraph):
    DG: nx.DiGraph = nx.DiGraph()
    target_edges = {}
    visited_nodes = set()
    for src, target, properties in MDG.edges(data=True):
        if (src, target) not in target_edges:
            target_edges[(src, target)] = properties
        else:
            if target_edges[(src, target)]["w_score"] < properties["w_score"]:
                target_edges[(src, target)] = properties

    _edges = []
    _properties = []
    for (src, target), p in target_edges.items():
        if src not in visited_nodes:
            visited_nodes.add(src)
            DG.add_node(src, index=src, name=src, type=MDG.nodes[src]['type'] if 'type' in MDG.nodes[src] else '')
        if target not in visited_nodes:
            visited_nodes.add(target)
            DG.add_node(target, index=target, name=target,
                        type=MDG.nodes[target]['type'] if 'type' in MDG.nodes[target] else '')
        _edges.append((src, target, p["w_score"]))
        _properties.append(p)

    for idx in range(len(_edges)):
        DG.add_edge(_edges[idx][0], _edges[idx][1], **_properties[idx])
    return DG


## 将一条路径转变为语句
def convert_a_path_to_sentence(path, DG: nx.MultiDiGraph):
    sentence = ""
    for edge in path.edges:
        # print(edge)
        originEdge = DG[edge.fromNode][edge.toNode]

        fromType = DG.nodes[edge.fromNode]['type']
        toType = DG.nodes[edge.toNode]['type']

        # 去除分裂时同一个实体产生的后缀
        _index = edge.toNode.rfind('_')
        originToNode = edge.toNode[:_index]

        relation = originEdge['relation_type']
        # sys_index = relation.find("sys_")
        # originRelation = relation[sys_index + 4:]

        if edge.fromNode == "V_source":
            sentence = sentence + toType + ":" + originToNode
            continue

        if edge.toNode == "V_sink":
            continue

        sentence = sentence + " " + relation + " " + toType + ":" + originToNode

    return sentence


def main():
    sql_engine = create_engine(db_config['uri'])
    # 0. 历史数据入库
    load_on_tsv('./thusword3new-opencart-auditd-0.tsv', save_db=True, sql_engine=sql_engine)

    # 1. 溯源图算分数、去环、找K-longest
    MDG = load_on_tsv('./thusword3new-opencart-auditd-0.tsv', save_db=False)
    add_source_and_sink(MDG)
    convertToDAG(MDG)
    print(MDG.number_of_nodes())

    # subax1 = plt.subplot(121)
    # options = {
    #     "node_color": "blue",
    #     "node_size": 1,
    #     "edge_color": "grey",
    #     "linewidths": 0,
    #     "width": 0.1,
    #     "arrows": False
    # }
    # nx.draw(MDG, **options)
    # plt.show()


def timer_check(prompt, timer):
    print(f"===== {prompt}: {time.time() - timer}")
    return time.time()


if __name__ == '__main__':
    timer = time.time()
    sql_engine = create_engine(db_config['uri'])

    # 0. 历史数据入库
    # load_on_tsv('./thusword3new-opencart-auditd-0.tsv', save_db=True, sql_engine=sql_engine)

    # 1. 打分
    MDG = load_on_tsv('./thusword3new-opencart-auditd-0.tsv', save_db=False)
    markRegularScore(MDG, sql_engine)
    timer = timer_check("Score", timer)
    # nx.write_gpickle(MDG, "./temp/scored_graph.pkl")

    # 2. 转换DAG
    add_source_and_sink(MDG)
    convertToDAG(MDG)
    timer = timer_check("Convert to DAG", timer)

    # 3. 选择K Longest Path
    DG = multi2Single(MDG)

    ctime = time.time()
    ksp_res = find_eppstein_ksp(DG, "V_source", "V_sink", 20, "w_score")
    timer = timer_check("K Longest Path", timer)

    # 4. 输出路径转换后的语句
    sentences = []

    for path in ksp_res:
        sentence = convert_a_path_to_sentence(path, DG)
        sentences.append(sentence)

    for sentence in sentences:
        print("*" * 100)
        print(sentence)
    timer = timer_check("Path to sentence", timer)

    # 5. 使用pv-dm对sentence进行嵌入
    pvdm_model = train_pvdm_model(sentences)

    embedded_sentences = []
    for sentence in sentences:
        embedded_sentence = convert_sentence_to_vector(sentence, pvdm_model)
        embedded_sentences.append(embedded_sentence)

    embedded_sentences = np.array(embedded_sentences)

    outliers, inliers = lof(embedded_sentences, k=5, method=1)
    for idx in outliers.index:
        print("Outliner--- " + sentences[idx])
    for idx in inliers.index:
        print("inliers--- " + sentences[idx])
    timer = timer_check("Outliner factor", timer)
