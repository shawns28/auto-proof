from auto_proof.code.pre import data_utils
from auto_proof.code.dataset import edge_list_to_adjency, prune_edges

import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import torch

def test(config):
    roots = data_utils.load_txt(config['data']['root_path'])   

    seed_index = config['loader']['seed_index']
    fov = config['loader']['fov']
    features_dir = config['data']['features_dir']
    map_pe_dir = config['data']['map_pe_dir']

    error_roots = []

    args_list = list([(root, features_dir, map_pe_dir, seed_index, fov) for root in roots])
    num_processes = 64
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for root in pool.imap_unordered(process_root, args_list):
            if root != 1111:
                error_roots.append(root)
            pbar.update()

    save_error_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/error_map_pe_roots.txt'
    data_utils.save_txt(save_error_path, error_roots)

def process_root(args):
    root, features_dir, map_pe_dir, seed_index, fov = args

    data_path = f'{features_dir}{root}.hdf5'
    map_pe_path = f'{map_pe_dir}map_{root}.hdf5'

    try:
        with h5py.File(data_path, 'r') as shard_file, h5py.File(map_pe_path, 'r') as map_pe_file:
            f = shard_file
            vertices = torch.from_numpy(f['vertices'][:])
            # radius = torch.from_numpy(f['radius'][:]).unsqueeze(1)
            # labels = torch.from_numpy(f['label'][:])
            # confidence = torch.from_numpy(f['confidence'][:])
            # labels = labels.unsqueeze(1).int()
            # confidence = confidence.unsqueeze(1).int()

            pos_enc = torch.from_numpy(map_pe_file['map_pe'][:])
            
            # rank_num = f'rank_{seed_index}'
            # rank = torch.from_numpy(f[rank_num][:])

            # size = len(vertices)

            # input = torch.cat((vertices, radius, pos_enc), dim=1)

            # edges = f['edges'][:]

            # if size > fov:
            #     indices = torch.where(rank < fov)[0]
            #     input = input[indices]
            #     labels = labels[indices]
            #     confidence = confidence[indices]
            #     edges = prune_edges(edges, indices)
            # elif(size < fov):
            #     input = torch.cat((input, torch.zeros((fov - size, input.shape[1]))), dim=0)
            #     labels = torch.cat((labels, torch.full((fov - size, 1), -1)))
            #     confidence = torch.cat((confidence, torch.full((fov - size, 1), -1)))

            # adj = edge_list_to_adjency(edges, fov)
            return 1111
    except Exception as e:
        print("root: ", root, "error: ", e)
        return root
                
if __name__ == "__main__":
    config = data_utils.get_config()
    test(config)
    