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

class Visitor(gt.BFSVisitor):
    def __init__(self, vp_rank):
        self.vp_rank = vp_rank
        self.rank = -1

    def examine_vertex(self, u):
        self.rank += 1
        self.vp_rank[u] = self.rank

def main(args):
    config, datastack_name, mat_version, client, data_directory = data_utils.initialize()
    # data_directory = '../../testing_data'
    # features_directory = f'{data_directory}/features_test'
    features_directory = f'{data_directory}/features'
    if not os.path.isdir(features_directory):
        os.makedirs(features_directory)

    root_id_to_rep_coords_path = f'{data_directory}/root_id_to_rep_coords_{mat_version}.pkl'
    root_id_to_rep_coords = data_utils.load_pickle_dict(root_id_to_rep_coords_path)

    chunk_num = int(args[0])
    num_chunks = 24
    # root_ids = data_utils.get_roots_chunk(chunk_num=chunk_num, num_chunks=num_chunks)

    # root_ids = data_utils.load_txt(f'{data_directory}/unprocessed_roots.txt')
    # print("len of roots", len(root_ids))
    
    cutoff = config["data"]["cutoff"]
    num_rand_seeds = config["data"]["num_rand_seeds"]

    # root_ids = [864691136272969918, 864691135776571232, 864691135474550843, 864691135445638290, 864691136899949422, 864691136175092486, 864691135937424949]
    # root_ids = [864691135445639826, 864691135446597204]
    root_ids = [864691131630728977, 864691131757470881]
    failed_roots = []
    with tqdm(total=len(root_ids)) as pbar:
        for root_id in root_ids:
            rep_coord = root_id_to_rep_coords[root_id]
            # print("root id:", root_id)
            # print("rep coord:", rep_coord)

            # print("Getting skel dict")
            retries = 2
            delay = 1
            failed = False
            for attempt in range(1, retries + 1):
                try: 
                    skel_dict = client.skeleton.get_skeleton(root_id=root_id, datastack_name=datastack_name, output_format='json')
                    break
                except Exception as e:
                    if attempt < retries:
                        print("Couldn't retrieve skeleton for root id: ", root_id, ". Attempt ", attempt + 1, "/", retries, "Error: ", e)
                        time.sleep(delay)
                        continue
                    else:
                        print("Failed to retrieve skeleton for root id: ", root_id, ". Skipping this root id") 
                        failed = True
                        failed_roots.append(root_id)
                        pbar.update()
                        continue
            
            if failed:
                continue

            # For testing dict
            # print("Saving json dict")
            # skel_json_path = f'{data_directory}/features_debugging/{root_id}.json'
            # data_utils.save_json(skel_json_path, skel_dict)

            try:
                skel_edges = np.array(skel_dict['edges'])
            except Exception as e:
                print("failed for root id: ", root_id, "error: ", e)
                failed_roots.append(root_id)
                pbar.update()
                continue
            skel_vertices = np.array(skel_dict['vertices'])
            skel_compartment = np.array(skel_dict['vertex_properties']['compartment'])
            skel_radii = np.array(skel_dict['vertex_properties']['radius'])

            # print("Getting closest")
            rep_index, _ = get_closest(skel_vertices, rep_coord)
            # print("rep_index", rep_index)

            #print("Getting ranks")
            try:
                g = gt.Graph(skel_edges, directed=False)
            except Exception as e:
                print("failed to create graph for root id: ", root_id, "error: ", e)
                failed_roots.append_(root_id)
                pbar.update()
                continue

            skel_len = len(skel_vertices)
            # print("Len of initial nodes ", skel_len)
            ranks = create_ranks(cutoff, g, rep_index, num_rand_seeds, skel_len)

            mask_indices = create_mask_indices(ranks, cutoff, num_rand_seeds)
            # print("mask_indices", mask_indices)
            # print("mask indices len", len(mask_indices))

            # Make ranks above cutoff max int as a kind of mask
            for i in range(len(ranks)):
                ranks[i] = ranks[i][mask_indices]
                ranks[i][ranks[i] >= cutoff] = np.iinfo(np.int32).max

            new_skel_vertices = skel_vertices[mask_indices]
            new_skel_compartment = skel_compartment[mask_indices]
            new_skel_radii = skel_radii[mask_indices]

            new_edges = prune_edges(mask_indices, skel_edges)
    
            # print("saving json after cutoff")
            # new_json_dict = {}
            # for i in range(len(ranks)):
            #     new_json_dict[f'rank_{i}'] = ranks[i].tolist()
            # data_utils.save_json(f'{root_dir}/{root_id}_{cutoff}.json', new_json_dict)

            # print("Saving h5")
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
            
            skel_hf_path = f'{features_directory}/{root_id}_{cutoff}.hdf5'
            data_utils.save_h5(skel_hf_path, features_dict)

            # print("loading h5")
            # with h5py.File(skel_hf_path, 'r') as f:
            #     vertices = f['vertices']
            #     edges = f['edges']
            #     rank_2 = f['rank_2']
            #     num_initial_vertices = f['num_initial_vertices']

            #     print("vertices", vertices[:])
            #     print("edges", edges[:])
            #     print("rank 2", rank_2[:])
            #     print("num_initial_vertices", num_initial_vertices[()])

            pbar.update()
    print("failed roots: ", failed_roots)
    
    # root_id = 864691136272969918
    # op_id = 107815

    # root_id = 864691135776571232
    # op_id = 47222

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
    # cutoff logical or mask
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

# Note that this will run bfs on the entire graph,
# won't do early termination which is ideal but don't want to figure it out right now
def bfs(g, seed_index):
    vp_rank = g.new_vp("int")
    gt.bfs_search(g, seed_index, Visitor(vp_rank))
    rank_arr = vp_rank.a
    # gt.graph_draw(g, vertex_text=vp_rank, output=f"graph-draw-{864691135776571232}-bfs.pdf")
    # Return should be a numpy array pretty sure
    return np.array(rank_arr)

def find_key():
    mat_version = 943
    data_directory = '../../data'
    input_dict = data_utils.load_pickle_dict(f'{data_directory}/operation_to_pre_edit_roots_{mat_version}.pkl')
    target = 864691136272969918
    op_id = next((key for key, value in input_dict.items() if value == target), None)
    print("op id", op_id)
    rep_dict = data_utils.load_pickle_dict(f'{data_directory}/operation_to_rep_coords_{mat_version}.pkl')
    print("rep coord", rep_dict[op_id])

def get_closest(arr, target):
    result = arr - target
    absit = np.abs(result)
    summed = absit.sum(axis=1)
    index = np.argmin(summed)
    value = arr[index]
    return index, value


if __name__ == "__main__":
    main(sys.argv[1:])