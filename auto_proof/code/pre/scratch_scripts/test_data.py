from auto_proof.code.pre import data_utils
from auto_proof.code.dataset import edge_list_to_adjency, prune_edges, adjency_to_edge_list_torch_skip_diag, adjency_to_edge_list_numpy_skip_diag

import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import torch
import time

def test(config):
    roots = data_utils.load_txt(config['data']['val_path'])
    # roots = [864691135937424949]
    # roots = [864691135778235581]

    # seed_index = config['loader']['seed_index']
    # print("seed index", seed_index)
    fov = config['loader']['fov']
    # print("fov", fov)
    features_dir = config['data']['features_dir']
    map_pe_dir = config['data']['map_pe_dir']
    box_cutoff = config['data']['box_cutoff']
    rank_dir = config['data']['rank_dir']
    max_cloud = config['trainer']['max_cloud']

    # error_roots = []
    # total_error_count = 0
    # total_conf_count = 0
    # total_node_count = 0
    # total_match_count = 0
    # total_error_match_count = 0
    # total_conf_smaller_than_cloud = 0

    conf_no_error_in_box = []

    # args_list = list([(root, features_dir, map_pe_dir, seed_index, fov, max_cloud) for root in roots])
    args_list = list([(root, features_dir, rank_dir, box_cutoff) for root in roots])
    num_processes = 32
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for root in pool.imap_unordered(process_root, args_list):
            if root != '':
                conf_no_error_in_box.append(root)
            # if root != 1111:
            #     error_roots.append(root)
            # total_node_count += node_count
            # total_error_count += error_count
            # total_conf_count += conf_count
            # total_match_count += match_count
            # total_error_match_count += error_match_count
            # total_conf_smaller_than_cloud += conf_smaller_than_cloud
            pbar.update()
    conf_no_error_in_box_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/conf_no_error_in_box_roots_val.txt'
    data_utils.save_txt(conf_no_error_in_box_path, conf_no_error_in_box)
    # print("total node count", total_node_count)
    # print("total error count", total_error_count)
    # print("total conf count", total_conf_count)
    # print("total match count", total_match_count)
    # print("total error match count", total_error_match_count)
    # print("total conf_smaller_than_cloud", total_conf_smaller_than_cloud)
    # print("total roots", len(roots))
    # save_error_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/error_map_pe_roots.txt'
    # data_utils.save_txt(save_error_path, error_roots)

def process_root(args):
    root, features_dir, rank_dir, box_cutoff = args
    # root, features_dir, map_pe_dir, seed_index, fov, max_cloud = args

    data_path = f'{features_dir}{root}.hdf5'
    # map_pe_path = f'{map_pe_dir}map_{root}.hdf5'
    rank_path = f'{rank_dir}rank_{root}.hdf5'

    try:
        # with h5py.File(data_path, 'r') as shard_file, h5py.File(map_pe_path, 'r') as map_pe_file:
        with h5py.File(data_path, 'r') as f, h5py.File(rank_path, 'r') as rank_file:
            # vertices = torch.from_numpy(f['vertices'][:])
            # radius = torch.from_numpy(f['radius'][:]).unsqueeze(1)
            labels = torch.from_numpy(f['label'][:])
            confidence = torch.from_numpy(f['confidence'][:])
            # labels = labels.unsqueeze(1).int()
            # confidence = confidence.unsqueeze(1).int()

            # pos_enc = torch.from_numpy(map_pe_file['map_pe'][:])

            rank = torch.from_numpy(rank_file[f'rank_{box_cutoff}'][:])

            size = len(labels)

            # input = torch.cat((vertices, radius, pos_enc), dim=1)

            # edges = f['edges'][:]

            # print("og edges", edges)

            if size > box_cutoff:
                indices = torch.where(rank < box_cutoff)[0]
                labels = labels[indices]
                confidence = confidence[indices]

            if torch.any(labels * confidence):
                return root
            else:
                return ''

            # if size > fov:
            #     indices = torch.where(rank < fov)[0]
            #     # input = input[indices]
            #     labels = labels[indices]
            #     confidence = confidence[indices]
                # edges = prune_edges(edges, indices)
            # elif(size < fov):
            #     input = torch.cat((input, torch.zeros((fov - size, input.shape[1]))), dim=0)
            #     labels = torch.cat((labels, torch.full((fov - size, 1), -1)))
            #     confidence = torch.cat((confidence, torch.full((fov - size, 1), -1)))

            # error_count = len(np.where(labels == 0)[0])
            # conf_count = len(np.where(confidence == 1)[0])
            # match_count = 0
            # error_match_count = 0
            # if error_count == conf_count:
            #     match_count = 1
            #     if error_count != 0:
            #         error_match_count = 1
            # conf_smaller_than_cloud = 0
            # if max_cloud - conf_count > 0:
            #     conf_smaller_than_cloud = 1
            # node_count = min(fov, size)

            # if abs(error_count - conf_count) == 1:
            #     print("c", conf_count, "e", error_count, "n", node_count, "root", root)

            # print("size", size)
            # time_1 = time.time()
            # adj = edge_list_to_adjency(edges, size, fov)
            # time_2 = time.time()
            # print("A normal", adj)
            # # print("A normal Time", time_2 - time_1)


            # time_5 = time.time()
            # edge_list = adjency_to_edge_list_torch_skip_diag(adj)
            # time_6 = time.time()
            # print("torch edges", edge_list)
            # print("torch edge time", time_6 - time_5)

            # time_7 = time.time()
            # edge_list = adjency_to_edge_list_numpy_skip_diag(adj)
            # time_8 = time.time()
            # print("numpy edges", edge_list)
            # print("numpy edge time", time_8 - time_7)

            # return node_count, error_count, conf_count, match_count, error_match_count, conf_smaller_than_cloud
    except Exception as e:
        print("root: ", root, "error: ", e)
        # return root
        return ''
        # return 0, 0, 0, 0, 0, 0
                
if __name__ == "__main__":
    config = data_utils.get_config()
    config['data']['map_pe_dir'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/map_pes2/"
    config['loader']['fov'] = 250
    test(config)
    