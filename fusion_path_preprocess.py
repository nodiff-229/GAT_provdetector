import json

from fusion_data import *

PP_ALL_BASE = "./data/tokenized/all"
PP_COMPRESS_BASE = "./data/tokenized/compressed"

PP_ALL_OUT_BASE = "./data/tokenized/all_pp"
PP_COMPRESS_OUT_BASE = "./data/tokenized/compressed_pp"

def preprocess(in_dir, out_dir):
    files = os.listdir(in_dir)
    all_rel_set = set()
    target_rel_set = {'read', 'write', 'delete', 'execute', 'append', 'fork', 'connect', 'bind', 'resolve', 'web_request',
                      'refer', 'executed', 'connected_remote_ip', 'sock_send', 'connected_session', 'resolve',
                      'send', 'receive', 'invoke', 'listen_remote_ip', 'transport_remote_ip', 'assign_remote_ip', 'down',
                      'redirect', 'addsubdirectory'}
    map_dict = {
        'accept': 'connected_remote_ip',
        'listen': 'listen_remote_ip',
        'transport': 'transport_remote_ip',
        'assignment': 'assign_remote_ip',
        "createpipeinstance": "create_pipe"
    }
    target_rel_set = target_rel_set.union(map_dict.values())

    for file in files:
        g = read_file(os.path.join(in_dir, file))
        raw_links = g['links']
        mod_links = []
        for raw_link in raw_links:
            for rel in raw_link['rel'].lower().split(","):
                if rel in map_dict:
                    rel = map_dict[rel]
                mod_link = raw_link.copy()
                mod_link['rel'] = rel
                mod_links.append(mod_link)
                all_rel_set.add(rel)
        with open(os.path.join(out_dir, file), 'w') as wf:
            json.dump({'nodes': g['nodes'], 'links': mod_links}, wf)

    print("intersection")
    print(all_rel_set.intersection(target_rel_set))

    print("in rel not in target")
    print(all_rel_set.difference(target_rel_set))

    print("in target not in rel")
    print(target_rel_set.difference(all_rel_set))

if __name__ == '__main__':
    preprocess(PP_ALL_BASE, PP_ALL_OUT_BASE)
    preprocess(PP_COMPRESS_BASE, PP_COMPRESS_OUT_BASE)