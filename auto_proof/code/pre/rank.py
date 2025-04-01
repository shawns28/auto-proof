from auto_proof.code.pre import data_utils

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import graph_tool.all as gt
import h5py
import json
import sys
import time
import argparse
import glob
import multiprocessing

'''
Custom class for graph tool to determine the rank using bfs
'''
class Visitor(gt.BFSVisitor):
    def __init__(self, vp_rank):
        self.vp_rank = vp_rank
        self.rank = -1

    def examine_vertex(self, u):
        self.rank += 1
        self.vp_rank[u] = self.rank

# Converts the currently saved rank 0 array to move the error to somewhere in the range of 100 and makes rank max 900
# NOTE: This won't work for the proofread roots that have not 000
def rank(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chunk_num", help="chunk num")
    parser.add_argument("-n", "--num_workers", help="num workers")
    args = parser.parse_args()
    chunk_num = 1 
    num_chunks = config['data']['num_chunks']
    if args.chunk_num:
        chunk_num = int(args.chunk_num)
    else: # No chunking
        num_chunks = 1
    if args.num_workers:
        config['loader']['num_workers'] = int(args.num_workers)

    roots = data_utils.load_txt(config['data']['root_path'])
    # roots = [864691135778235581]
    # roots = [864691134941129571]
    # roots = [864691135778235581, 864691134941129571]
    roots = data_utils.get_roots_chunk(config, roots, chunk_num=chunk_num, num_chunks=num_chunks)

    proofread_roots = data_utils.load_txt(config['data']['proofread_at_mat_path'])
    features_dir = config['data']['features_dir']
    rank_dir = config['data']['rank_dir']

    box_cutoff = config['data']['box_cutoff']

    num_processes = config['loader']['num_workers']
    args_list = list([(root, features_dir, rank_dir, box_cutoff, root in proofread_roots) for root in roots])

    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root, args_list):
            pbar.update()

def process_root(args):
    try:
        root, features_dir, rank_dir, box_cutoff, is_proofread = args
        data_path = f'{features_dir}{root}_000.hdf5'
        rank_path = f'{rank_dir}rank_{root}_000.hdf5'

        if os.path.exists(rank_path):
            return

        with h5py.File(data_path, 'r') as f:
            edges = f['edges'][:]
            rank_0 = f['rank_0'][:]
            if is_proofread:
                with h5py.File(rank_path, 'a') as rank_f:
                    rank_f.create_dataset(f'rank_{box_cutoff}', data=rank_0)
                    return

            # Create the graph from the edges
            # Make the seed index the current 0 in the rank 0
            # Run the bfs until you get a rank that has the rep index inside of 100

            g = gt.Graph(edges, directed=False)

            rep_index = np.where(rank_0 == 0)[0]

            rep_included = False
            while not rep_included:
                seed = np.random.randint(len(rank_0))
                new_rank = bfs(g, seed)
                if new_rank[rep_index] < box_cutoff:
                    rep_included = True

            with h5py.File(rank_path, 'a') as rank_f:
                rank_f.create_dataset(f'rank_{box_cutoff}', data=new_rank)
    except Exception as e:
        print("error with root", root, "error", e)
        return
                

# NOTE: This will run bfs on the entire graph and won't do early termination which would be faster
def bfs(g, seed_index):
    vp_rank = g.new_vp("int")
    gt.bfs_search(g, seed_index, Visitor(vp_rank))
    rank_arr = vp_rank.a
    return np.array(rank_arr)

if __name__ == "__main__":
    config = data_utils.get_config()
    rank(config)