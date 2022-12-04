from lemmatization_utils import *

if __name__ == '__main__':
    with open(f"./GNN_data_integrate/node_check.txt", 'w') as ff:
        hosts = ['S1', 'S2', 'S3', 'S4', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
        for host in hosts:
            ff.write("====== Host ======\n")
            with open(f"./GNN_data_integrate/{host}.json", 'r') as f:
                target_obj = json.load(f)
            for node in target_obj['nodes']:
                anony_str = ','.join(node['anonymous'])
                nid = node['id']

                # if ('.com/' in nid or '.net/' in nid or '.org/' in nid or '.com:9999' in nid) \
                #     and (anony_str == 'web_object' or anony_str == 'url'
                #          or anony_str == 'web_object,url' or anony_str == 'url,web_object'):
                #     continue

                if (is_url(nid)) \
                    and (anony_str == 'web_object' or anony_str == 'url'
                         or anony_str == 'web_object,url' or anony_str == 'url,web_object'):
                    continue

                if anony_str == 'IP_Address' and is_ip(nid):
                    continue

                if '.exe_' in nid and 'process' in anony_str:
                    continue

                ff.write(f"{','.join(node['anonymous'])}\t{node['id']}\n")