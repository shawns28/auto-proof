from auto_proof.code.pre import data_utils

import numpy as np
from tqdm import tqdm
import h5py
import graph_tool.all as gt
import sys
import multiprocessing
from collections import deque
import random
import argparse
import os
import glob
import time

def main():
    """
    TODO: Convert this entire file to be networkx
    """
    data_config = data_utils.get_config('data')
    client_config = data_utils.get_config('client')
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_processes", help="num processes")
    args = parser.parse_args()
    if args.num_processes:
        data_config['multiprocessing']['num_processes'] = int(args.num_processes)

    data_dir = data_config['data_dir']
    client, datastack_name, mat_version_start, mat_version_end = data_utils.create_client(client_config)

    roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'
    features_dir = f'{data_dir}{data_config['features']['features_dir']}'
    labels_dir = f'{data_dir}{data_config['labels']['labels_at_latest_dir']}{data_config['labels']['latest_mat_version']}/'

    post_label_roots = data_utils.load_txt(f'{roots_dir}{data_config['labels']['post_label_roots']}')
    print("post_label_roots len", len(post_label_roots))
    # TODO: Uncomment once segclr is done
    # post_segclr_roots = data_utils.load_txt(f'{roots_dir}{data_config['segclr']['post_segclr_roots']}')
    # print("post_segclr_roots len", len(post_segclr_roots))
    # roots = np.intersect1d(post_label_roots, post_segclr_roots)
    # print("roots len", len(roots))
    roots = post_label_roots
    roots_1300_unique_copied = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/1300_unique_copied.txt")
    roots = np.setdiff1d(roots, roots_1300_unique_copied)
    print("roots len", len(roots))
    split = data_config['split']['split']
    split_dir = f'{roots_dir}{data_config['split']['split_dir']}{len(roots)}/'
    split_dicts_dir = f'{split_dir}dicts/'
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    if not os.path.exists(split_dicts_dir):
        os.makedirs(split_dicts_dir)

    num_processes = data_config['multiprocessing']['num_processes']

    root_to_latest_set_path = f'{split_dicts_dir}root_to_latest_set.pkl'
    if not os.path.exists(root_to_latest_set_path):
        print("creating root to latest set")
        roots_at_latest_dir = f'{data_dir}{data_config['labels']['roots_at_latest_dir']}{data_config['labels']['latest_mat_version']}/'
        root_to_latest_set = create_latest_set_dict(roots, roots_at_latest_dir, num_processes, root_to_latest_set_path)
        print("root to latest set key len", len(root_to_latest_set.keys()))
    else:
        print("loading root to latest set")
        root_to_latest_set = data_utils.load_pickle_dict(root_to_latest_set_path)
    
    print("creating graph")
    g, vmap = create_graph(root_to_latest_set)
    print("len of vertices from graph", len(g.get_vertices()))

    comp, _ = gt.label_components(g)
    comp_arr = np.array(comp.a)

    root_latest_to_cc_path = f'{split_dicts_dir}root_latest_to_cc.pkl'
    if not os.path.exists(root_latest_to_cc_path):
        print("Creating root_latest_to_cc dict")
        root_latest_to_cc = create_root_latest_to_cc(vmap, comp_arr, root_latest_to_cc_path)
    else:
        print("Loading root_latest_to_cc dict")
        root_latest_to_cc = data_utils.load_pickle_dict(root_latest_to_cc_path)

    root_to_cc_path = f'{split_dicts_dir}root_to_cc.pkl'
    if not os.path.exists(root_to_cc_path):
        print("Creating root_to_cc dict")
        root_to_cc = create_root_to_cc(root_latest_to_cc, root_to_latest_set, root_to_cc_path)
        print("Number of roots in root_to_cc", len(root_to_cc.keys()))
    else:
        print("Loading root_to_cc dict")
        root_to_cc = data_utils.load_pickle_dict(root_to_cc_path)

    cc_to_root_group_path = f'{split_dicts_dir}cc_to_root_group.pkl'
    if not os.path.exists(cc_to_root_group_path):
        print("Creating cc_to_root_group dict")
        cc_to_root_group = create_cc_to_root_group(root_to_cc, cc_to_root_group_path)
    else:
        print("Loading cc_to_root_group dict")
        cc_to_root_group = data_utils.load_pickle_dict(cc_to_root_group_path)
    
    cc_root_group_list_path = f'{split_dicts_dir}cc_root_group_list.pkl'
    if not os.path.exists(cc_root_group_list_path):
        print("Creating cc_root_group_list")
        cc_root_group_list = create_cc_root_group_list(cc_to_root_group, cc_root_group_list_path)
    else:
        print("Loading cc_root_group_list")
        cc_root_group_list = data_utils.load_pickle_dict(cc_root_group_list_path)

    proofread_mat_version1 = data_config['proofread']['mat_versions'][0]
    proofread_mat_version2 = data_config['proofread']['mat_versions'][1]
    proofread_roots = data_utils.load_txt(f'{data_dir}{data_config['proofread']['proofread_dir']}{proofread_mat_version1}_{proofread_mat_version2}_with_copies.txt')
    
    val_roots_path = f'{split_dir}val_roots.txt'
    if not os.path.exists(val_roots_path):
        print("Creating split")
        data_utils.save_txt(f'{split_dir}all_roots.txt', roots)
        start_time = time.time()
        train_roots, val_roots, test_roots = create_split(cc_root_group_list, proofread_roots, split, split_dir)
        end_time = time.time()
        print("Creating split took", end_time - start_time)
        
        print("Checking split")
        check_proofread_dist(proofread_roots, train_roots, val_roots, test_roots)
    else:
        print("Loading val roots")
        val_roots = data_utils.load_txt(val_roots_path)
    
    val_conf_no_error_in_box_roots_path = f'{split_dir}val_conf_no_error_in_box_roots.txt'
    if not os.path.exists(val_conf_no_error_in_box_roots_path):
        print("Creating val_conf_no_error_in_box_roots")
        create_conf_no_error_in_box_roots(val_roots, features_dir, labels_dir, data_config['features']['box_cutoff'], num_processes, val_conf_no_error_in_box_roots_path)

def create_latest_set_dict(roots, roots_at_latest_dir, num_processes, save_path):
    root_to_latest_set = {}
    args_list = [(root, roots_at_latest_dir) for root in roots]
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for latest_roots, root in pool.imap_unordered(__get_latest_roots__, args_list):
            root_to_latest_set[root] = set(latest_roots)
            pbar.update()
    data_utils.save_pickle_dict(save_path, root_to_latest_set)
    return root_to_latest_set

def __get_latest_roots__(data):
    root, roots_at_latest_dir = data
    roots_at_path = f'{roots_at_latest_dir}{root}.hdf5'
    with h5py.File(roots_at_path, 'r') as roots_at_f:
        latest_roots = roots_at_f['roots_at'][:]
    return latest_roots, root

def create_graph(root_to_latest_set):
    g = gt.Graph(directed=False)
    edges = []
    # all_vertices = []
    for root in root_to_latest_set:
        curr_list = list(root_to_latest_set[root])
        if len(curr_list) == 1:
            vertice = str(curr_list[0])
            edges.append((vertice, vertice))
            # all_vertices.append(vertice)
            # g.add_edge_list([(vertice, vertice)], hashed=True, hash_type="string")
            # print("root with one latest root: root is ", root, "latest root is ", vertice)
        else:
            pairs = [(str(curr_list[i]), str(curr_list[i + 1])) for i in range(len(curr_list) - 1)]
            # curr_vertices = [str(curr_list[i]) for i in range(len(curr_list))]
            # all_vertices.extend(curr_vertices)
            edges.extend(pairs)
            # g.add_edge_list(pairs, hashed=True, hash_type="string")
    
    # print("all unique vertices", len(all_vertices) == len(set(all_vertices)))
    # print("len of unique vertices", len(set(all_vertices)))
    vmap = g.add_edge_list(edges, hashed=True, hash_type="string")

    return g, vmap

def create_root_latest_to_cc(vmap, comp_arr, root_latest_to_cc_path):
    root_latest_to_cc = {}
    for i in range(len(comp_arr)):
        root_latest_to_cc[int(vmap[i])] = comp_arr[i]
    data_utils.save_pickle_dict(root_latest_to_cc_path, root_latest_to_cc)
    return root_latest_to_cc

def create_root_to_cc(root_latest_to_cc, root_to_latest_set, root_to_cc_path):
    root_to_cc = {}
    for root in root_to_latest_set:
        root_latest = root_to_latest_set[root].pop()
        root_to_cc[root] = root_latest_to_cc[root_latest]
    data_utils.save_pickle_dict(root_to_cc_path, root_to_cc)
    return root_to_cc

def create_cc_to_root_group(root_to_cc, cc_to_root_group_path):
    cc_to_root_group = {}
    for root in root_to_cc:
        cc = root_to_cc[root]
        if cc not in cc_to_root_group:
            cc_to_root_group[cc] = []
        cc_to_root_group[cc].append(root)
    data_utils.save_pickle_dict(cc_to_root_group_path, cc_to_root_group)
    return cc_to_root_group

def create_cc_root_group_list(cc_to_root_group, cc_root_group_list_path):
    cc_root_group_list = []
    for cc in cc_to_root_group:
        cc_root_group_list.append(cc_to_root_group[cc])
    # print("Printing first 3 groups", cc_root_group_list[:3])
    data_utils.save_pickle_dict(cc_root_group_list_path, cc_root_group_list)
    return cc_root_group_list

def create_split(cc_root_group_list, proofread_roots, split, split_dir):    
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
                break;

    print("Final proportions for proofread/size", sum_proofread_proportions, sum_size_proportions)
    print("Split roots len", len(split_roots[0]), len(split_roots[1]), len(split_roots[2]))

    print(np.intersect1d(np.array(proofread_roots), np.array(split_roots[0])).size)

    data_utils.save_txt(f'{split_dir}train_roots.txt', split_roots[0])
    data_utils.save_txt(f'{split_dir}val_roots.txt', split_roots[1])
    data_utils.save_txt(f'{split_dir}test_roots.txt', split_roots[2])
    with open(f'{split_dir}split_metadata', 'w') as meta_f:
        meta_f.write(f'train_{sum_size_proportions[0]}_{sum_proofread_proportions[0]}\n')
        meta_f.write(f'val_{sum_size_proportions[1]}_{sum_proofread_proportions[1]}\n')
        meta_f.write(f'test_{sum_size_proportions[2]}_{sum_proofread_proportions[2]}\n')
    return np.array(split_roots[0]), np.array(split_roots[1]), np.array(split_roots[2])

def check_proofread_dist(proofread_roots, train_roots, val_roots, test_roots):
    train_count = np.intersect1d(proofread_roots, train_roots).size
    val_count = np.intersect1d(proofread_roots, val_roots).size
    test_count = np.intersect1d(proofread_roots, test_roots).size

    print("proofread count in train/val/test", train_count, val_count, test_count)
    print("proofread ratio in train/val/test", train_count/len(proofread_roots), val_count/len(proofread_roots), test_count/len(proofread_roots))

def create_conf_no_error_in_box_roots(roots, features_dir, labels_dir, box_cutoff, num_processes, conf_no_error_in_box_roots_path):
    args_list = list([(root, features_dir, labels_dir, box_cutoff) for root in roots])
    conf_no_error_in_box = []
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for root in pool.imap_unordered(__check_root__, args_list):
            if root != '':
                conf_no_error_in_box.append(root)
            pbar.update()
    data_utils.save_txt(conf_no_error_in_box_roots_path, conf_no_error_in_box)

def __check_root__(data):
    root, features_dir, labels_dir, box_cutoff = data
    feature_path =  f'{features_dir}{root}.hdf5'
    labels_path = f'{labels_dir}{root}.hdf5'
    with h5py.File(feature_path, 'r') as feat_f, h5py.File(labels_path, 'r') as labels_f:
        labels = labels_f['labels'][:]
        confidences = labels_f['confidences'][:]
        rank = feat_f['rank'][:]
        size = len(labels)
        if size > box_cutoff:
            indices = np.where(rank < box_cutoff)[0]
            labels = labels[indices]
            confidences = confidences[indices]

        if np.any((1 - labels) * confidences): 
            return root
        else:
            return ''


if __name__ == "__main__":
    main()
    
