from auto_proof.code.pre import data_utils

from caveclient import CAVEclient
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

'''
Skeletonizes the roots and saves hdf5 files representing the cutoff number of nodes and their features
which are pulled from cave client for the root. Additionally saves successful roots to txt.
Input:
    config
Flags:
    -c, --chunk_num: This will chunk the root_ids for multi node runs if provided
'''
def skeletonize(config):
    # TODO: Make this compatible with running the script in pre_process_main.py
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chunk_num", help="chunk num")
    args = parser.parse_args()
    chunk_num = 1
    num_chunks = config['data']['num_chunks']
    if args.chunk_num:
        chunk_num = int(args.chunk_num)
    else: # No chunking
        num_chunks = 1

    root_ids = data_utils.load_txt(config['data']['root_path'])
    root_ids = data_utils.get_roots_chunk(config, root_ids, chunk_num=chunk_num, num_chunks=num_chunks)
    # root_ids = data_utils.load_txt(f'{data_directory}root_ids/unprocessed_roots.txt')    
    # root_ids = [864691136272969918, 864691135776571232, 864691135474550843, 864691135445638290, 864691136899949422, 864691136175092486, 864691135937424949]
    # root_ids = [864691135445639826, 864691135446597204]
    # root_ids = [864691131630728977, 864691131757470881]

    is_proofread = config['data']['is_proofread']

    client, datastack_name, mat_version = data_utils.create_client(config)
    data_directory = config['data']['data_dir']
    features_directory = config['data']['features_dir']

    if not os.path.isdir(features_directory):
        os.makedirs(features_directory)

    root_id_to_rep_coords_path = f'{data_directory}dicts/root_id_to_rep_coords_{mat_version}.pkl'
    root_id_to_rep_coords = data_utils.load_pickle_dict(root_id_to_rep_coords_path)

    cutoff = config["data"]["cutoff"]
    num_rand_seeds = config["data"]["num_rand_seeds"]    
    
    with tqdm(total=len(root_ids)) as pbar:
        for root_id in root_ids:

            skel_hf_path = f'{features_directory}{root_id}.hdf5'
            # Skip already processed roots
            if os.path.exists(skel_hf_path):
                pbar.update()
                continue

            # NOTE: It would seem that the skeleton service has changed so skeletonizing proofread vs pre-edit
            # would have been different code. Check commit history for previous version.
            retries = 2
            delay = 1
            failed = False
            for attempt in range(1, retries + 1):
                try: 
                    skel_dict = client.skeleton.get_skeleton(root_id=root_id, datastack_name=datastack_name, output_format='dict')
                    break
                except Exception as e:
                    if attempt < retries:
                        print("Couldn't retrieve skeleton for root id: ", root_id, ". Attempt ", attempt, "/", retries, "Error: ", e)
                        time.sleep(delay)
                        continue
                    else:
                        print("Failed to retrieve skeleton for root id: ", root_id, ". Skipping this root id") 
                        failed = True
                        pbar.update()
                        continue
            
            if failed:
                continue

            # TODO: Delete after running
            # For testing dict
            # print("Saving json dict")
            # skel_json_path = f'{data_directory}{root_id}.json'
            # data_utils.save_json(skel_json_path, skel_dict)

            try:
                skel_edges = np.array(skel_dict['edges'])
                skel_vertices = np.array(skel_dict['vertices'])
                skel_compartment = np.array(skel_dict['compartment'])
                skel_radii = np.array(skel_dict['radius'])
            except Exception as e:
                print("failed for root id: ", root_id, "error: ", e)
                pbar.update()
                continue

            # ALL OF THE RANK RELATED LOGIC IS DEPRECATED
            # TODO: update the code so that it works better for everything
            rep_index = np.random.randint(0, len(skel_vertices))
            if not is_proofread:
                rep_coord = root_id_to_rep_coords[root_id]
                rep_index, _ = get_closest(skel_vertices, rep_coord)
                print("rep_coord", rep_coord)
                print("rep_index", rep_index)
                print("vertices", skel_vertices)

            try:
                g = gt.Graph(skel_edges, directed=False)
            except Exception as e:
                print("failed to create graph for root id: ", root_id, "error: ", e)
                pbar.update()
                continue

            skel_len = len(skel_vertices)
            ranks = create_ranks(cutoff, g, rep_index, num_rand_seeds, skel_len)
            mask_indices = create_mask_indices(ranks, cutoff, num_rand_seeds)

            for i in range(len(ranks)):
                ranks[i] = ranks[i][mask_indices]
                ranks[i][ranks[i] >= cutoff] = np.iinfo(np.int32).max

            new_skel_vertices = skel_vertices[mask_indices]
            new_skel_compartment = skel_compartment[mask_indices]
            new_skel_radii = skel_radii[mask_indices]

            new_edges = prune_edges(mask_indices, skel_edges)
    
            # TODO: Delete after testing
            # print("saving json after cutoff")
            # new_json_dict = {}
            # for i in range(len(ranks)):
            #     new_json_dict[f'rank_{i}'] = ranks[i].tolist()
            # data_utils.save_json(f'{root_dir}/{root_id}_{cutoff}.json', new_json_dict)

            features_dict = {
                'root_id': root_id,
                'num_initial_vertices': len(skel_vertices),
                'num_vertices': len(new_skel_vertices),
                'cutoff': cutoff,
                'vertices': new_skel_vertices,
                'edges': new_edges,
                'compartment': new_skel_compartment,
                'radius': new_skel_radii
            }
            
            for i in range(len(ranks)):
                features_dict[f'rank_{i}'] = ranks[i]
            
            # Save features to hdf5 file
            with h5py.File(skel_hf_path, 'a') as skel_hf:
                for feature in features_dict:
                    skel_hf.create_dataset(feature, data=features_dict[feature])
            pbar.update()

def prune_edges(mask_indices, skel_edges):
    orig_to_new = {value: index for index, value in enumerate(mask_indices)}
    edge_mask = np.zeros(len(skel_edges))
    for i in range(len(skel_edges)):
        edge = skel_edges[i]
        if edge[0] in orig_to_new and edge[1] in orig_to_new:
            skel_edges[i][0] = orig_to_new[edge[0]]
            skel_edges[i][1] = orig_to_new[edge[1]]
            edge_mask[i] = 1
    return skel_edges[edge_mask.astype(bool)]

def create_mask_indices(ranks, cutoff, num_rand_seeds):
    mask = ranks[0] < cutoff
    for i in range(num_rand_seeds):
        curr_mask = ranks[i + 1] < cutoff
        mask = np.logical_or(mask, curr_mask)
    return np.where(mask == 1)[0]

def create_ranks(cutoff, g, rep_index, num_rand_seeds, skel_len):
    ranks = []
    rank_arr = bfs(g, rep_index)
    ranks.append(rank_arr)
    for _ in range(num_rand_seeds):
        rep_included = False
        while not rep_included:
            seed = np.random.randint(skel_len)
            rank_arr = bfs(g, seed)
            if rank_arr[rep_index] < cutoff:
                rep_included = True
                ranks.append(rank_arr)
    return ranks

# NOTE: This will run bfs on the entire graph and won't do early termination which would be faster
def bfs(g, seed_index):
    vp_rank = g.new_vp("int")
    gt.bfs_search(g, seed_index, Visitor(vp_rank))
    rank_arr = vp_rank.a
    return np.array(rank_arr)

def get_closest(arr, target):
    result = arr - target
    absit = np.abs(result)
    summed = absit.sum(axis=1)
    index = np.argmin(summed)
    value = arr[index]
    return index, value

def save_skeletonized_roots(config, post_skel_path):
    files = glob.glob(f'{config['data']['features_dir']}*')
    roots = [files[i][-23:-5] for i in range(len(files))]
    print(roots[0])
    data_utils.save_txt(post_skel_path, roots)
    
if __name__ == "__main__":
    config = data_utils.get_config()
    # config['data']['is_proofread'] = True
    # config['data']['features_dir'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/proofread_features/"
    # config['data']['root_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/proofread_943.txt"
    # post_skel_path = f'{config['data']['data_dir']}root_ids/post_skel_proofread_roots.txt'

    config['data']['is_proofread'] = False
    config['data']['features_dir'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/features/"
    config['data']['root_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/debugging_roots.txt"
    post_skel_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/post_skel_debugging_roots.txt"

    skeletonize(config)
    
    # NOTE: This will save the file each time a chunk finishes but should still result in the correct file
    # print("Saving successfuly skeletonized roots list as txt")
    # save_skeletonized_roots(config, post_skel_path)