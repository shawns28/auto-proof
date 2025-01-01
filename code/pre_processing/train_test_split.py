import data_utils
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import json
import fastremap
import sys
import time
import matplotlib.pyplot as plt
import glob
import graph_tool.all as gt
from collections import defaultdict
import multiprocessing


def serial(): 
    files = glob.glob('../../data/successful_labels/*')
    roots = [files[i][-18:] for i in range(len(files))]
    root_to_943_set = defaultdict(set)
    with tqdm(total=len(roots)) as pbar:
        for root in roots:
            root_path = f'../../data/features/{root}_1000.h5py'
            with h5py.File(root_path, 'r') as f:
                root_943s = f['root_943'][:]
            for root_943 in root_943s:
                root_to_943_set[root_943].add(root)
            pbar.update()
    data_utils.save_pickle_dict('../../data/root_to_943_set.pkl', root_to_943_set)
    print("key len", len(root_to_943_set.keys()))
    print("val len", len(root_to_943_set.values()))

def parallel():
    files = glob.glob('../../data/successful_labels/*')
    roots = np.array([files[i][-18:] for i in range(len(files))])
    # roots = roots[:500]
    manager = multiprocessing.Manager()
    # root_to_943_set = manager.dict()
    root_to_943_set = {}
    num_processes = 64
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for root_943s, root in pool.imap_unordered(__process_operation__, roots):
            for root_943 in root_943s:
                if not root_943 in root_to_943_set:
                    root_to_943_set[root_943] = set()
                root_to_943_set[root_943].add(root)
                # print(root_to_943_set[root_943])
                # print(root)
            pbar.update()
    data_utils.save_pickle_dict('../../data/root_to_943_set.pkl', root_to_943_set)
    print("key len", len(root_to_943_set.keys()))
    print("val len", len(root_to_943_set.values()))
    # print(root_to_943_set)

def remap():
    files = glob.glob('../../data/successful_labels/*')
    print(len(files))
    root_to_943_set = data_utils.load_pickle_dict('../../data/root_to_943_set.pkl')
    print("key len", len(root_to_943_set.keys()))
    roots = data_utils.load_txt('../../data/post_label_roots_459972.txt')
    for root_943 in 

def create_graph():
    g = gt.Graph(directed=False)


    root_to_943_set = data_utils.load_pickle_dict('../../data/root_to_943_set.pkl')
    print("key len", len(root_to_943_set.keys()))
    # for root_943 in root_to_943_set:
    #     curr_list = list(root_to_943_set[root_943])
    #     # print(curr_list)

    #     # if len(curr_list == 1):
    #     #     g.add_vertex
    #     pairs = [(int(curr_list[i]), int(curr_list[i + 1])) for i in range(len(curr_list) - 1)]
    #     g.add_edge_list(pairs)
    #     break
    g.add_edge(864691134186128802, 864691134186128803)
    print(g.get_edges())


def __process_operation__(data):
    root = data
    root_path = f'../../data/features/{root}_1000.h5py'
    with h5py.File(root_path, 'r') as f:
        root_943s = f['root_943'][:]
    return root_943s, root

if __name__ == "__main__":
    # parallel()
    # create_graph()
    remap()

