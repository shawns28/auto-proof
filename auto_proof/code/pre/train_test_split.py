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
import re

def main():
    """
    TODO: Convert this entire file to be networkx

    Algorithm: 
    1. root to "root at latest". Ex: 1 -> 112, 112, 113, etc
    2. create a graph where shared "roots at latest" get edges connnecting them. 
        Ex: 1 -> {112, 113} and 2 -> {112, 114} becomes a graph(113 <-> 112 <-> 114)
    3. Get the connected components so idenfier of component to component. Ex: 5 -> graph(113 <-> 112 <-> 114)
    4. root latest to cc identifier. Ex: 113 -> 5, 112 -> 5, 114 -> 5
    5. root to cc. Ex: 1 -> 5, 2 -> 5
    6. cc to root group. 5 -> 1, 2
    7. root group list [[1, 2]]

    Modified version:
    1. add in a "root at latest" to root. Ex: 112 -> (1, 2), 113 -> 1, 114 -> 2 - This actually should happen after since it will be specific to the roots
    2. get the largest cc, could just call largest component graph tool method
    3. calculate the betweennes similarity on this component
    4. Sort the "roots at latest" based off of the betweennes score
    5. For the top k "roots at latest", use the "roots at latest" to root dict to get the roots that would need to be removed.
    6. Remove the roots from the graph and recompute the components, run the largest component again to see what the size is now
    7. modify the roots latest to cc to use a root list and reference the dict instead with the k roots removed

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
    
    post_segclr_roots = data_utils.load_txt(f'{roots_dir}{data_config['segclr']['post_segclr_roots']}')
    print("post_segclr_roots len", len(post_segclr_roots))
    roots = np.intersect1d(post_label_roots, post_segclr_roots)
    print("roots combined len", len(roots))
    roots_1300_unique_copied = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/1300_unique_copied.txt")
    roots = np.setdiff1d(roots, roots_1300_unique_copied)
    print("final roots len", len(roots))

    # TODO: Remove after
    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/all_roots_og.txt")
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/new_and_prev_shared_roots.txt")

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
    
    # Going to try making the train test split logic based off of the box root logic
    no_error_in_box_dict_path = f'{split_dicts_dir}no_error_in_box_dict.pkl'
    if not os.path.exists(no_error_in_box_dict_path):
        print("Creating no error in box dict")
        no_error_in_box_dict = create_conf_no_error_in_box_roots(roots, features_dir, labels_dir, data_config['features']['box_cutoff'], num_processes, no_error_in_box_dict_path)
    else:
        print("Loading no error in box dict")
        no_error_in_box_dict = data_utils.load_pickle_dict(no_error_in_box_dict_path)

    print("creating graph")
    g, vmap = create_graph(root_to_latest_set)
    print("len of vertices from graph", len(g.get_vertices()))

    comp, _ = gt.label_components(g)
    comp_arr = np.array(comp.a)

    # 1. add in a "root at latest" to root. Ex: 112 -> (1, 2), 113 -> 1, 114 -> 2 - This actually should happen after since it will be specific to the roots
    # 2. get the largest cc, could just call largest component graph tool method
    # 3. calculate the betweennes similarity on this component
    # 4. Sort the "roots at latest" based off of the betweennes score
    # 5. For the top k "roots at latest", use the "roots at latest" to root dict to get the roots that would need to be removed.
    # 6. Remove the roots from the graph and recompute the components, run the largest component again to see what the size is now
    # 7. modify the roots latest to cc to use a root list and reference the dict instead with the k roots removed

    # largest = gt.label_largest_component(g)
    # largest_arr_indices = np.array(largest.a).astype(bool)

    # largest_latest_roots = []
    # for i in range(len(largest_arr_indices)):
    #     if largest_arr_indices[i]:
    #         largest_latest_roots.append(int(vmap[i]))
    
    # print("len of largest component", len(largest_latest_roots))

    # subgraph = gt.GraphView(g, vfilt=largest)
    # print("subgraph vertices", subgraph.num_vertices())


    # betweenness_path = f'{split_dicts_dir}betweennes.npy'
    # if not os.path.exists(betweenness_path):
    #     vertex_betweennes, _ = gt.betweenness(g)
    #     print("Saving betweennes of largest component")
    #     np.save(betweenness_path, vertex_betweennes.a)
    #     betweenness = vertex_betweennes.a
    # else:
    #     print("Loading betweennes")
    #     betweenness = np.load(betweenness_path)
    # print(betweenness[:10])

    # largest_latest_to_betweenness = {}
    # for i, root in enumerate(largest_latest_roots):
    #     largest_latest_to_betweenness[root] = betweenness[i]

    # largest_latest_to_betweenness = dict(sorted(largest_latest_to_betweenness.items(), key=lambda item: item[1], reverse=True))

    # Get the root at latest to the roots for everything and save it
    # For each root at latest in the top k, create a list of all the roots
    # create a subdirectory
    # redo basically all the methods
    # latest_to_roots = {}
    # for root in root_to_latest_set:
    #     latest_set = root_to_latest_set[root]
    #     for latest_root in latest_set:
    #         if latest_root not in latest_to_roots:
    #             latest_to_roots[latest_root] = set()
    #         latest_to_roots[latest_root].add(root)
    
    # print("len of latest_to_roots", len(latest_to_roots))
    # count = 0
    # roots_to_remove = set()
    # for key in largest_latest_to_betweenness:
    #     # print(key, largest_latest_to_betweenness[key])
    #     # print(key, latest_to_roots[key])
    #     roots_to_remove.update(latest_to_roots[key])
    #     count += 1
    #     if count == 50000:
    #         break

    # Going to try something else since this didn't work
    # bicomp, art_points, nc  = gt.label_biconnected_components(subgraph)
    # articulation_vertices = [int(vmap[v]) for v in subgraph.vertices() if art_points[v]]
    # print(articulation_vertices[0])
    # print("len of articulation points", len(articulation_vertices))
    # print("number of biconnected components", nc)
    # print("unsorted art", articulation_vertices[:5])
    # sorted_articulation_vertices = sorted(articulation_vertices, key=lambda item: largest_latest_to_betweenness[item])
    # print("sorted art", sorted_articulation_vertices[:5])
        
    # count = 0
    # roots_to_remove = set()
    # for latest_root in sorted_articulation_vertices:
    #     # print(key, largest_latest_to_betweenness[key])
    #     # print(key, latest_to_roots[key])
    #     roots_to_remove.update(latest_to_roots[latest_root])
    #     count += 1
    #     if count == 650:
    #         break

    # Trying just removing the roots with lots of roots at latest
    # count = 0
    # roots_to_remove = set()
    # sorted_root_to_latest = dict(sorted(root_to_latest_set.items(), key=lambda item: len(item[1])))
    # for root in sorted_root_to_latest:
    #     roots_to_remove.add(root)
    #     count += 1
    #     if count == 50000:
    #         break


    # Trying to just remove the first 50000 to see if that does it
    # roots_to_remove = roots[:110000]

    # print("len of roots to remove", len(roots_to_remove))
    # roots = np.setdiff1d(roots, np.array(list(roots_to_remove)))
    # print("len of roots after removing", len(roots))

    # for root in roots_to_remove:
    #     root_to_latest_set.pop(root)

    # split_dicts_dir = f'{split_dir}dicts_{len(roots)}/'
    # if not os.path.exists(split_dicts_dir):
    #     os.makedirs(split_dicts_dir)

    # print("creating graph")
    # g, vmap = create_graph(root_to_latest_set)
    # print("len of vertices from graph 2", len(g.get_vertices()))

    # comp, _ = gt.label_components(g)
    # comp_arr = np.array(comp.a)

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

    # cc_root_group_list.sort(key=len, reverse=True)
    # count = 0
    # for cc_group in cc_root_group_list:
    #     print("size", len(cc_group))
    #     count += 1
    #     if count == 10:
    #         break

    proofread_mat_version1 = data_config['proofread']['mat_versions'][0]
    proofread_mat_version2 = data_config['proofread']['mat_versions'][1]
    proofread_roots = data_utils.load_txt(f'{data_dir}{data_config['proofread']['proofread_dir']}{proofread_mat_version1}_{proofread_mat_version2}.txt')
    # proofread_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_unique.txt")
    proofread_roots = [str(root) + '_000' for root in proofread_roots]

    # TODO: REMEMBER TO CHANGE THE DIR if doing the special stuff
    split_dicts_dir = f'{roots_dir}{data_config['split']['split_dir']}{len(roots)}/'
    val_roots_path = f'{split_dicts_dir}val_roots_hi.txt'
    train_roots_path = f'{split_dicts_dir}train_roots_hi.txt'
    test_roots_path = f'{split_dicts_dir}test_roots_hi.txt'
    if not (os.path.exists(val_roots_path) and os.path.exists(train_roots_path)):
        print("Creating split")
        data_utils.save_txt(f'{split_dicts_dir}all_roots_hi.txt', roots)
        start_time = time.time()
        default = True
        train_roots, val_roots, test_roots = create_split(no_error_in_box_dict, cc_root_group_list, proofread_roots, split, split_dicts_dir, default)
        end_time = time.time()
        print("Creating split took", end_time - start_time)
        
        print("Checking split")
        check_proofread_dist(proofread_roots, train_roots, val_roots, test_roots)
    else:
        print("Loading val and train roots")
        val_roots = data_utils.load_txt(val_roots_path)
        train_roots = data_utils.load_txt(train_roots_path)
        test_roots = data_utils.load_txt(test_roots_path)
    
    # TODO: Remake the actual create conf roots
    val_conf_no_error_in_box_roots_list_path = f'{split_dicts_dir}val_conf_no_error_in_box_roots_hi.txt'
    train_conf_no_error_in_box_roots_list_path = f'{split_dicts_dir}train_conf_no_error_in_box_roots_hi.txt'
    test_conf_no_error_in_box_roots_list_path = f'{split_dicts_dir}test_conf_no_error_in_box_roots_hi.txt'
    if not (os.path.exists(val_conf_no_error_in_box_roots_list_path) and os.path.exists(train_conf_no_error_in_box_roots_list_path)):
        print("Creating val_conf_no_error_in_box_roots")
        create_conf_no_error_in_box_roots_list(no_error_in_box_dict, val_roots, val_conf_no_error_in_box_roots_list_path)
        print("Creating train_conf_no_error_in_box_roots")
        create_conf_no_error_in_box_roots_list(no_error_in_box_dict, train_roots, train_conf_no_error_in_box_roots_list_path)
        print("Creating test_conf_no_error_in_box_roots")
        create_conf_no_error_in_box_roots_list(no_error_in_box_dict, test_roots, test_conf_no_error_in_box_roots_list_path)

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

def create_split(no_error_in_box_dict, cc_root_group_list, proofread_roots, split, split_dir, default):    
    # def count_shared_elements(cc_root_group):
    #     return np.intersect1d(proofread_roots, np.array(cc_root_group)).size
    # root_list = sorted(cc_root_group_list, key=count_shared_elements, reverse=True)
    
    if default:
        root_list = sorted(
        cc_root_group_list,
        key=lambda sublist: np.intersect1d(proofread_roots, np.array(sublist)).size,
        reverse=True # Added to sort by descending count (more proofread roots first)
    ) 
    else:
        # The new version that I should make into a flag or mention in the docs and remove
        root_list = sorted(
            cc_root_group_list,
            key=lambda sublist: sum(1 for root in sublist if no_error_in_box_dict.get(root, False)),
            # Use .get(root, False) for robustness if 'root' might not be in the dictionary.
            # If you are absolutely sure 'root' will always be a key, you can use:
            # key=lambda sublist: sum(1 for root in sublist if no_error_in_box_dict[root]),
            # Add reverse=True if you want to sort by descending count (more Trues first)
            # reverse=True
        )

    # conf_count_list = [count_true_in_sublist(sub_roots, no_error_in_box_dict) for sub_roots in root_list]
    # mini_count = 0
    # for sub_roots in cc_root_group_list:
    #     print("len of sub roots", len(sub_roots))
    #     print("number of conf", sum(1 for root in sub_roots if no_error_in_box_dict.get(root, False)))
    #     mini_count += 1
    #     if mini_count == 50:
    #         break

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
    # split_roots[0].extend(root_list.popleft())
    # while len(root_list) > 0:
        

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

    for sub_split_roots in split_roots:
        print("sub_split_roots conf count", sum(1 for root in sub_split_roots if no_error_in_box_dict.get(root, False)))

    print(np.intersect1d(np.array(proofread_roots), np.array(split_roots[0])).size)

    data_utils.save_txt(f'{split_dir}train_roots_hi.txt', split_roots[0])
    data_utils.save_txt(f'{split_dir}val_roots_hi.txt', split_roots[1])
    data_utils.save_txt(f'{split_dir}test_roots_hi.txt', split_roots[2])
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
    conf_no_error_in_box = {}
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for root, root_bool in pool.imap_unordered(__check_root__, args_list):
            conf_no_error_in_box[root] = root_bool
            pbar.update()
    data_utils.save_pickle_dict(conf_no_error_in_box_roots_path, conf_no_error_in_box)
    return conf_no_error_in_box

def create_conf_no_error_in_box_roots_list(conf_no_error_in_box, roots, conf_no_error_in_box_roots_list_path):
    conf_list = []
    for root in roots:
        if conf_no_error_in_box[root]:
            conf_list.append(root)
    
    print("conf_list len before filtering", len(conf_list))
    filtered_roots = [root for root in conf_list if re.search(r'_000$', root)]
    print("conf_list len after filtering", len(filtered_roots))
    data_utils.save_txt(conf_no_error_in_box_roots_list_path, filtered_roots)

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
            return root, True
        else:
            return root, False

if __name__ == "__main__":
    main()
    
