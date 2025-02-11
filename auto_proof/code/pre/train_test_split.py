from auto_proof.code.pre import data_utils

import numpy as np
from tqdm import tqdm
import h5py
import graph_tool.all as gt
import sys
import multiprocessing
from collections import deque
import random

# Create the root 943 set in parallel
def create_943_set_dict(config):
    print("creating root to 943 set")
    roots = data_utils.load_txt(config['data']['root_path'])
    root_paths = [f'{config['data']['features_dir']}{root}.hdf5' for root in roots]
    root_to_943_set = {}
    num_processes = config['loader']['num_workers']
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for root_943s, root in pool.imap_unordered(__process_operation__, root_paths):
            root_to_943_set[root] = set(root_943s)
            pbar.update()
    data_utils.save_pickle_dict(f'{config['data']['data_dir']}dicts/root_to_943_set.pkl', root_to_943_set)
    print("key len", len(root_to_943_set.keys()))
    print("val len", len(root_to_943_set.values()))
    return root_to_943_set

def __process_operation__(root_path):
    root = root_path[-23:-5]
    with h5py.File(root_path, 'r') as f:
        root_943s = f['root_943'][:]
    return root_943s, root

def create_graph(root_to_943_set):
    print("creating graph")
    g = gt.Graph(directed=False)
    edges = []
    # all_vertices = []
    for root in root_to_943_set:
        curr_list = list(root_to_943_set[root])
        if len(curr_list) == 1:
            vertice = str(curr_list[0])
            edges.append((vertice, vertice))
            # all_vertices.append(vertice)
            # g.add_edge_list([(vertice, vertice)], hashed=True, hash_type="string")
            # print("root with one 943 root: root is ", root, "943 root is ", vertice)
        else:
            pairs = [(str(curr_list[i]), str(curr_list[i + 1])) for i in range(len(curr_list) - 1)]
            curr_vertices = [str(curr_list[i]) for i in range(len(curr_list))]
            # all_vertices.extend(curr_vertices)
            edges.extend(pairs)
            # g.add_edge_list(pairs, hashed=True, hash_type="string")
    
    # print("all unique vertices", len(all_vertices) == len(set(all_vertices)))
    # print("len of unique vertices", len(set(all_vertices)))
    vmap = g.add_edge_list(edges, hashed=True, hash_type="string")
    print("len of vertices from graph", len(g.get_vertices()))

    return g, vmap

def create_root_943_to_cc(config, vmap, comp_arr):
    print("Creating root_943_to_cc")
    root_943_to_cc = {}
    for i in range(len(comp_arr)):
        root_943_to_cc[int(vmap[i])] = comp_arr[i]
    data_utils.save_pickle_dict(f'{config['data']['data_dir']}dicts/root_943_to_cc.pkl', root_943_to_cc)
    return root_943_to_cc

def create_root_to_cc(config, root_943_to_cc, root_to_943_set):
    print("Creating root_to_cc dict")
    root_to_cc = {}
    for root in root_to_943_set:
        root_943 = root_to_943_set[root].pop()
        root_to_cc[root] = root_943_to_cc[root_943]
    print("Number of roots in root_to_cc", len(root_to_cc.keys()))
    data_utils.save_pickle_dict(f'{config['data']['data_dir']}dicts/root_to_cc.pkl', root_to_cc)
    return root_to_cc

def create_cc_to_root_group(config, root_to_cc):
    print("Creating cc_to_root dict")
    cc_to_root_group = {}
    for root in root_to_cc:
        cc = root_to_cc[root]
        if cc not in cc_to_root_group:
            cc_to_root_group[cc] = []
        cc_to_root_group[cc].append(root)
    data_utils.save_pickle_dict(f'{config['data']['data_dir']}dicts/cc_to_root_group.pkl', cc_to_root_group)
    return cc_to_root_group

def create_cc_root_group_list(config, cc_to_root_group):
    print("Creating cc_root_group_list")
    cc_root_group_list = []
    for cc in cc_to_root_group:
        cc_root_group_list.append(cc_to_root_group[cc])
    # print("Printing first 3 groups", cc_root_group_list[:3])
    data_utils.save_pickle_dict(f'{config['data']['data_dir']}dicts/cc_root_group_list.pkl', cc_root_group_list)
    return cc_root_group_list

def create_split(cc_root_group_list, split, split_dir):
    proofread_roots = data_utils.load_txt(config['data']['proofread_at_mat_path'])
    
    def count_shared_elements(cc_root_group):
        return np.intersect1d(proofread_roots, np.array(cc_root_group)).size
    root_list = sorted(cc_root_group_list, key=count_shared_elements, reverse=True)

    size_list = [len(arr) for arr in root_list]
    proofread_count_list = [np.intersect1d(proofread_roots, np.array(cc_root_group)).size for cc_root_group in root_list]
    print("First 15 proofread count", proofread_count_list[:15])
    print("First 15 size list", size_list[:15])
    print("Sample of group len", len(root_list[0]), len(root_list[1]), len(root_list[2]), len(root_list[10]), len(root_list[-1]))
    
    root_list = deque(root_list)
    size_list = deque(size_list)
    proofread_count_list = deque(proofread_count_list)

    sum_size_proportions = [0 for _ in split]
    sum_proofread_proportions = [0 for _ in split]
    split_roots = [[] for _ in split]

    # First make sure we can do the best we can for proofread roots and then distribute the rest to make total proportions

    while len(root_list) > 0 and proofread_count_list[0] > 0:
        split_roots[0].extend(root_list.popleft())
        sum_size_proportions[0] += size_list.popleft()
        sum_proofread_proportions[0] += proofread_count_list.popleft()

        while (len(root_list) > 0 and proofread_count_list[0] > 0 and sum_proofread_proportions[1] / sum_proofread_proportions[0]) * (split[0] / split[1]) < 1 and (sum_proofread_proportions[2] / sum_proofread_proportions[0]) * (split[0] / split[2]) < 1:
            if len(root_list) and proofread_count_list[0] > 0:
                for i in range(1, len(sum_proofread_proportions)):
                    if len(root_list) > 0 and proofread_count_list[0] > 0:
                        split_roots[i].extend(root_list.popleft())
                        sum_size_proportions[i] += size_list.popleft()
                        sum_proofread_proportions[i] += proofread_count_list.popleft()
                    else:
                        break
            else:
                break
    
    print("Split roots len after proofread distributed", len(split_roots[0]), len(split_roots[1]), len(split_roots[2]))

    while len(root_list) > 0:
        split_roots[0].extend(root_list.popleft())
        sum_size_proportions[0] += size_list.popleft()
        sum_proofread_proportions[0] += proofread_count_list.popleft()

        while (len(root_list) > 0 and sum_size_proportions[1] / sum_size_proportions[0]) * (split[0] / split[1]) < 1 and (sum_size_proportions[2] / sum_size_proportions[0]) * (split[0] / split[2]) < 1:
            if len(root_list) > 0:
                for i in range(1, len(sum_size_proportions)):
                    if len(root_list) > 0:
                        split_roots[i].extend(root_list.popleft())
                        sum_size_proportions[i] += size_list.popleft()
                        sum_proofread_proportions[i] += proofread_count_list.popleft()
                    else:
                        break
            else:
                break

    print("Final proportions for proofread/size", sum_proofread_proportions, sum_size_proportions)
    print("Split roots len", len(split_roots[0]), len(split_roots[1]), len(split_roots[2]))

    print(np.intersect1d(np.array(proofread_roots), np.array(split_roots[0])).size)

    data_utils.save_txt(f'{split_dir}train_roots_{sum_size_proportions[0]}_{sum_proofread_proportions[0]}.txt', split_roots[0])
    data_utils.save_txt(f'{split_dir}val_roots_{sum_size_proportions[1]}_{sum_proofread_proportions[1]}.txt', split_roots[1])
    data_utils.save_txt(f'{split_dir}test_roots_{sum_size_proportions[2]}_{sum_proofread_proportions[2]}.txt', split_roots[2])
    return np.array(split_roots[0]), np.array(split_roots[1]), np.array(split_roots[2])

def check_proofread_dist(config, train_roots, val_roots, test_roots):
    proofread_roots = data_utils.load_txt(config['data']['proofread_at_mat_path'])
    train_count = np.intersect1d(proofread_roots, train_roots).size
    val_count = np.intersect1d(proofread_roots, val_roots).size
    test_count = np.intersect1d(proofread_roots, test_roots).size

    print("proofread count in train/val/test", train_count, val_count, test_count)
    print("proofread ratio in train/val/test", train_count/len(proofread_roots), val_count/len(proofread_roots), test_count/len(proofread_roots))


if __name__ == "__main__":
    config = data_utils.get_config()
    # config['data']['data_dir'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/"
    # config['data']['root_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/post_label_roots_459972.txt"
    # root_to_943_set = create_943_set_dict(config)
    # root_to_943_set = data_utils.load_pickle_dict(f'{config['data']['data_dir']}dicts/root_to_943_set.pkl')
    
    # g, vmap = create_graph(root_to_943_set)

    # comp, _ = gt.label_components(g)
    # comp_arr = np.array(comp.a)

    # root_943_to_cc = create_root_943_to_cc(config, vmap, comp_arr)
    # root_to_cc = create_root_to_cc(config, root_943_to_cc, root_to_943_set)
    # cc_to_root_group = create_cc_to_root_group(config, root_to_cc)
    
    # cc_root_group_list = create_cc_root_group_list(config, cc_to_root_group)
    cc_root_group_list = data_utils.load_pickle_dict(f'{config['data']['data_dir']}dicts/cc_root_group_list.pkl')
    # cc_root_group_list = data_utils.load_pickle_dict(f'{config['data']['data_dir']}dicts/cc_roots_500.pkl')
    # cc_root_group_list = data_utils.load_pickle_dict(f'{config['data']['data_dir']}/debugging_data/dicts/cc_root_group_list.pkl')

    split = (0.8, 0.1, 0.1)
    split_dir = f'{config['data']['data_dir']}root_ids/'
    train_roots, val_roots, test_roots = create_split(cc_root_group_list, split, split_dir)

    # train_roots = data_utils.load_txt(config['data']['train_path'])
    # val_roots = data_utils.load_txt(config['data']['val_path'])
    # test_roots = data_utils.load_txt(config['data']['test_path'])

    check_proofread_dist(config, train_roots, val_roots, test_roots)
    
