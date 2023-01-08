"""
read, write, delete, execute, fork, connect, bind, resolve, web_request, refer, executed, connected_remote_ip, sock_send, connected_session

"""
import json
import os.path
import re
import networkx as nx
from commons.embedding import write_sentences_to_file

def is_url(s):
    if re.match(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
                s) is not None:
        return True
    return False


def is_ip(s):
    s = s.split('_')[0]
    if re.match(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", s) is not None \
            or re.match(
        r"(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))",
        s) is not None:
        return True
    else:
        return False


def tokenize_sequences_one(seq_list):
    i = 0
    seq_list = seq_list.copy()
    if seq_list[i * 3 + 1] == "source edge" or seq_list[i * 3 + 1] == "sink edge":
        return None

    if seq_list[i * 3 + 1] == "read" or seq_list[i * 3 + 1] == "write" or seq_list[i * 3 + 1] == "delete" or \
            seq_list[i * 3 + 1] == "execute" or seq_list[i * 3 + 1] == "append" or \
            seq_list[i * 3 + 1] == "addsubdirectory":
        if "c:/windows/system32" in seq_list[i * 3]:
            seq_list[i * 3] = "system32_process"
        elif "c:/windows" in seq_list[i * 3]:
            seq_list[i * 3] = "windows_process"
        elif "c:/programfiles" in seq_list[i * 3]:
            seq_list[i * 3] = "programfiles_process"
        elif "c:/users" in seq_list[i * 3]:
            seq_list[i * 3] = "user_process"
        else:
            seq_list[i * 3] = "process"

        if not ";" in seq_list[i * 3 + 2]:
            if "c:/windows/system32" in seq_list[i * 3 + 2]:
                seq_list[i * 3 + 2] = "system32_file"
            elif "c:/windows" in seq_list[i * 3 + 2]:
                seq_list[i * 3 + 2] = "windows_file"
            elif "c:/programfiles" in seq_list[i * 3 + 2]:
                seq_list[i * 3 + 2] = "programfiles_file"
            elif "c:/users" in seq_list[i * 3 + 2]:
                seq_list[i * 3 + 2] = "user_file"
            else:
                seq_list[i * 3 + 2] = "file"
        else:
            seq_list[i * 3 + 2] = "combined_files"
    elif seq_list[i * 3 + 1] == "fork" or seq_list[i * 3 + 1] == "create_pipe":
        if "c:/windows/system32" in seq_list[i * 3]:
            seq_list[i * 3] = "system32_process"
        elif "c:/windows" in seq_list[i * 3]:
            seq_list[i * 3] = "windows_process"
        elif "c:/programfiles" in seq_list[i * 3]:
            seq_list[i * 3] = "programfiles_process"
        elif "c:/users" in seq_list[i * 3]:
            seq_list[i * 3] = "user_process"
        else:
            seq_list[i * 3] = "process"

        if "c:/windows/system32" in seq_list[i * 3 + 2]:
            seq_list[i * 3 + 2] = "system32_process"
        elif "c:/windows" in seq_list[i * 3 + 2]:
            seq_list[i * 3 + 2] = "windows_process"
        elif "c:/programfiles" in seq_list[i * 3 + 2]:
            seq_list[i * 3 + 2] = "programfiles_process"
        elif "c:/users" in seq_list[i * 3 + 2]:
            seq_list[i * 3 + 2] = "user_process"
        else:
            seq_list[i * 3 + 2] = "process"
    elif seq_list[i * 3 + 1] == "connect" or seq_list[i * 3 + 1] == "bind":
        if "c:/windows/system32" in seq_list[i * 3]:
            seq_list[i * 3] = "system32_process"
        elif "c:/windows" in seq_list[i * 3]:
            seq_list[i * 3] = "windows_process"
        elif "c:/programfiles" in seq_list[i * 3]:
            seq_list[i * 3] = "programfiles_process"
        elif "c:/users" in seq_list[i * 3]:
            seq_list[i * 3] = "user_process"
        else:
            seq_list[i * 3] = "process"

        if seq_list[i * 3 + 1] == "connect":
            seq_list[i * 3 + 2] = "connection"  # "IP_Address"
        else:
            seq_list[i * 3 + 2] = "session"
    elif seq_list[i * 3 + 1] == "resolve":
        seq_list[i * 3] = "IP_Address"
        seq_list[i * 3 + 2] = "domain_name"
    elif seq_list[i * 3 + 1] == "web_request":
        seq_list[i * 3] = "domain_name"
        seq_list[i * 3 + 2] = "web_object"
    elif seq_list[i * 3 + 1] == "refer":
        seq_list[i * 3] = "web_object"
        seq_list[i * 3 + 2] = "web_object"
    elif seq_list[i * 3 + 1] == "executed":
        if "c:/windows/system32" in seq_list[i * 3]:
            seq_list[i * 3] = "system32_file"
        elif "c:/windows" in seq_list[i * 3]:
            seq_list[i * 3] = "windows_file"
        elif "c:/programfiles" in seq_list[i * 3]:
            seq_list[i * 3] = "programfiles_file"
        elif "c:/users" in seq_list[i * 3]:
            seq_list[i * 3] = "user_file"
        else:
            seq_list[i * 3] = "file"

        if "c:/windows/system32" in seq_list[i * 3 + 2]:
            seq_list[i * 3 + 2] = "system32_process"
        elif "c:/windows" in seq_list[i * 3 + 2]:
            seq_list[i * 3 + 2] = "windows_process"
        elif "c:/programfiles" in seq_list[i * 3 + 2]:
            seq_list[i * 3 + 2] = "programfiles_process"
        elif "c:/users" in seq_list[i * 3 + 2]:
            seq_list[i * 3 + 2] = "user_process"
        else:
            seq_list[i * 3 + 2] = "process"
    elif seq_list[i * 3 + 1] == "sock_send":
        seq_list[i * 3] = "session"
        seq_list[i * 3 + 2] = "session"
    elif seq_list[i * 3 + 1] == "listen_remote_ip" or seq_list[i * 3 + 1] == "assign_remote_ip" \
            or (seq_list[i * 3 + 1] == "transport_remote_ip" and is_ip(seq_list[i * 3 + 2])):
        seq_list[i * 3 + 2] = "IP_Address"
        if not seq_list[i * 3].startswith("connection_"):
            if "c:/windows/system32" in seq_list[i * 3]:
                seq_list[i * 3] = "system32_process"
            elif "c:/windows" in seq_list[i * 3]:
                seq_list[i * 3] = "windows_process"
            elif "c:/programfiles" in seq_list[i * 3]:
                seq_list[i * 3] = "programfiles_process"
            elif "c:/users" in seq_list[i * 3]:
                seq_list[i * 3] = "user_process"
            else:
                seq_list[i * 3] = "process"
        else:
            seq_list[i * 3] = "connection"
    elif seq_list[i * 3 + 1] == "connected_remote_ip" \
            or (seq_list[i * 3 + 1] == "transport_remote_ip" and is_ip(seq_list[i * 3])):
        seq_list[i * 3] = "IP_Address"
        if not seq_list[i * 3 + 2].startswith("connection_"):
            if "c:/windows/system32" in seq_list[i * 3 + 2]:
                seq_list[i * 3 + 2] = "system32_process"
            elif "c:/windows" in seq_list[i * 3 + 2]:
                seq_list[i * 3 + 2] = "windows_process"
            elif "c:/programfiles" in seq_list[i * 3 + 2]:
                seq_list[i * 3 + 2] = "programfiles_process"
            elif "c:/users" in seq_list[i * 3 + 2]:
                seq_list[i * 3 + 2] = "user_process"
            else:
                seq_list[i * 3 + 2] = "process"
        else:
            seq_list[i * 3 + 2] = "connection"
    elif seq_list[i * 3 + 1] == "connected_session":
        seq_list[i * 3] = "IP_Address"
        seq_list[i * 3 + 2] = "session"
    elif seq_list[i * 3 + 1] == "down":
        if re.match(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", seq_list[i * 3]) is not None:
            seq_list[i * 3] = "IP_Address"
        else:
            seq_list[i * 3] = "file"
        seq_list[i * 3 + 2] = "url"
    elif seq_list[i * 3 + 1] == "redirect":
        seq_list[i * 3] = "domain"
        seq_list[i * 3 + 2] = "url"
    return seq_list


def parse_graph(host, path='./final/audit'):
    scored_dg = nx.read_gpickle(f"{path}/{host}_Scored.pkl")
    with open(f"{path}/{host}_NodeLabel.json", 'r') as f:
        nlabels = json.load(f)
    pos_nodes = set(nlabels['pos'])
    neg_nodes = set(nlabels['neg'])

    parsed_list = []
    node_dict = {}

    for edge in scored_dg.edges(data=True):
        src, target, data = edge
        # if src == "system_3" and target == "192.168.223.128_1":
        #     print()
        rela = data['relation_type']
        time = data['time']
        w_score = data['w_score']

        parsed = tokenize_sequences_one([src, rela, target])
        if parsed is not None:
            anoy_src, _, anoy_target = parsed
            info = {
                "src": (src, anoy_src),
                "target": (target, anoy_target),
                "rela": rela,
                "time": time,
                "w_score": w_score
            }
            clean_src = re.findall(r"^(.*)_\d+$", src)[0]
            clean_target = re.findall(r"^(.*)_\d+$", target)[0]
            if src not in node_dict:
                assert clean_src in pos_nodes or clean_src in neg_nodes
                node_dict[src] = {'malicious': clean_src in pos_nodes, 'anonymous': set()}
            node_dict[src]['anonymous'].add(anoy_src)

            if target not in node_dict:
                assert clean_target in pos_nodes or clean_target in neg_nodes
                node_dict[target] = {'malicious': clean_target in pos_nodes, 'anonymous': set()}
            node_dict[target]['anonymous'].add(anoy_target)
            parsed_list.append(info)

    node_list = []
    for node_id, node_data in node_dict.items():
        node_data['id'] = node_id
        node_data['anonymous'] = list(node_data['anonymous'])
        node_list.append(node_data)

    return {'nodes': node_list, 'edges': parsed_list}


def lemma_preprocess(path='./final/audit'):
    hosts = ['S1', 'S2', 'S3', 'S4', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    # hosts = ['M5']
    out_dir = os.path.join(path, 'json_result')
    os.makedirs(out_dir, exist_ok=True)
    for host in hosts:
        out = parse_graph(host, path)
        with open(os.path.join(out_dir, f"{host}.json"), 'w') as f:
            json.dump(out, f)

if __name__ == '__main__':
    lemma_preprocess(path='./final/audit')
    lemma_preprocess(path='./final/auditcpr')
    lemma_preprocess(path='./final/nodemerge')