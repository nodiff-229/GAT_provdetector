import datetime as dt
import json
import os
import time
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine

from main import insert_sql, reduce_len

NODE_TYPES = {
    0: "user", 1: "process", 2: "socket", 3: "obj"
}
DIR_BASE = "./data/tokenized/all_pp"

with open("./db.json", 'r') as db_config_f:
    db_config = json.load(db_config_f)


def read_file(filepath):
    with open(filepath, "r") as f:
        g = json.load(f)
    return g

def date_summary():
    files = os.listdir(DIR_BASE)
    sort_list = []
    node_dict = {}

    for file in files:
        g = read_file(os.path.join(DIR_BASE, file))
        for node in g['nodes']:
            node_dict[reduce_len(node['id'])] = {
                # 'node_type': NODE_TYPES[node["type"]],
                'name': reduce_len(node['name']),
                'malicious': not node['malicious_label'] == 'no'
            }

        for link in g['links']:
            ts = link['ts']
            t = time.localtime(ts)
            minute = t.tm_min // 5
            dt = time.strftime("%Y-%m-%d %H", t)
            dt = f"{file}:{dt}:{'%02d' % (minute * 5)}"
            link["data_src"] = file
            # start, end, time, relation_type, success = row.start, row.end, row.time, row.relation_type, row.success
            new_link = {
                "start": reduce_len(link["source"]), "end": reduce_len(link["target"]), "relation_type": link["rel"],
                "time": int(link["ts"] * 1000), "success": True, "host": file.replace(f"-audit.json", "")
            }
            sort_list.append(new_link)
    sort_list = sorted(sort_list, key=lambda lk: lk['time'])
    return sort_list, node_dict


def split(sort_list, slice=1000):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(sort_list), slice):
        print(f"{str(i)}/{str(len(sort_list))}")
        yield sort_list[i:i + slice]


def load_db(chunk, node_dict, sql_engine, timewindow=None):
    # 每批数据
    df = pd.DataFrame(chunk)
    in_process = []
    out_process = []
    for row in df.itertuples():
        line = row.Index
        start, end, time, relation_type, success, host = row.start, row.end, row.time, row.relation_type, row.success, row.host

        if not (len(str(time)) == 10 or len(str(time)) == 13):
            raise Exception("时间戳长度不为10或13")
        if timewindow is None:
            date_str = datetime.fromtimestamp(time if len(str(time)) == 10 else time / 1000).strftime("%Y-%m-%d")
        else:
            date_str = timewindow
        in_process.append([end, date_str, host, str(relation_type)])
        out_process.append([start, date_str, host, str(relation_type)])

    insert_sql(in_process, out_process, df, sql_engine)


if __name__ == '__main__':
    sort_list, node_dict = date_summary()
    date_stub = datetime(2022, 1, 1)
    sql_engine = create_engine(db_config['uri'])
    for chunk in split(sort_list, slice=10000):
        print(date_stub)
        load_db(chunk, node_dict, sql_engine, date_stub.strftime("%Y-%m-%d"))
        date_stub = date_stub + dt.timedelta(days=1)