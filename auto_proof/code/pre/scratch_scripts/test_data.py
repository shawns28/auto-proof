from auto_proof.code.pre import data_utils
from auto_proof.code.dataset import edge_list_to_adjency, prune_edges, adjency_to_edge_list_torch_skip_diag, adjency_to_edge_list_numpy_skip_diag

import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import torch
import time
import glob

def test():
    data_config = data_utils.get_config('data')
    data_dir = data_config['data_dir']
    roots_dir = f'{data_config['data_dir']}roots_{343}_{1300}/'
    latest_version = data_config['labels']['latest_mat_version']
    labels_at_latest_dir = f'{data_dir}{data_config['labels']['labels_at_latest_dir']}{latest_version}/'
    roots_at_latest_dir = f'{data_dir}{data_config['labels']['roots_at_latest_dir']}{latest_version}/'

    # roots = data_utils.load_txt(f'{roots_dir}{data_config['features']['post_feature_roots']}')
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_503534/all_roots.txt")
    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_503534/val_conf_no_error_in_box_roots.txt")
    proofread_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_1300_copied.txt")

    no_error_roots = []
    # args_list = list([(root, features_dir, map_pe_dir, seed_index, fov, max_cloud) for root in roots])
    args_list = list([(root, data_config) for root in roots])
    num_processes = 64
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for no_error_root in pool.imap_unordered(process_root, args_list):
            if no_error_root != '':
                no_error_roots.append(no_error_root)
            pbar.update()
            
    print("total counts", len(roots))

    result = np.setdiff1d(no_error_roots, proofread_roots)

    print("total", len(result))
    print("total - proofread roots len", len(result) - len(proofread_roots))


def process_root(args):
    root, data_config = args

    feature_path = f'{data_config['data_dir']}{data_config['features']['features_dir']}{root}.hdf5'
    latest_version = data_config['labels']['latest_mat_version']
    labels_at_latest_path = f'{data_config['data_dir']}{data_config['labels']['labels_at_latest_dir']}{latest_version}/{root}.hdf5'

    try:
        
        with h5py.File(labels_at_latest_path, 'r') as labels_f, h5py.File(feature_path, 'r') as feat_f:
            # vertices = torch.from_numpy(f['vertices'][:])
            # radius = torch.from_numpy(f['radius'][:]).unsqueeze(1)
            labels = labels_f['labels'][:]
            # labels = labels.unsqueeze(1).int()
            # confidence = confidence.unsqueeze(1).int()
            size = len(labels)
            box_cutoff = 100
            rank = feat_f['rank'][:]

            if size > box_cutoff:
                indices = np.where(rank < box_cutoff)[0]
                labels = labels[indices]

            if np.any(labels > 0):
                return ''
            else:
                return root

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

    test()
    