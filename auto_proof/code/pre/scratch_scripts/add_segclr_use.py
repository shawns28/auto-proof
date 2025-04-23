from auto_proof.code.pre import data_utils
from auto_proof.code.dataset import edge_list_to_adjency, prune_edges, adjency_to_edge_list_torch_skip_diag, adjency_to_edge_list_numpy_skip_diag

import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import torch
import time
import os
import glob

def test():
    data_config = data_utils.get_config('data')
    data_dir = data_config['data_dir']
    roots_dir = f'{data_config['data_dir']}roots_{343}_{1300}/'
    latest_version = data_config['labels']['latest_mat_version']
    labels_at_latest_dir = f'{data_dir}{data_config['labels']['labels_at_latest_dir']}{latest_version}/'
    roots_at_latest_dir = f'{data_dir}{data_config['labels']['roots_at_latest_dir']}{latest_version}/'

    # roots = data_utils.load_txt(f'{roots_dir}{data_config['features']['post_feature_roots']}')
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_379668/all_roots.txt")
    files = glob.glob(f'{"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/segclr/"}*')
    roots = [files[i][-27:-5] for i in range(len(files))]
    # roots = ['864691134940888163_000']
    print("roots len", len(roots))
    # proofread_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_1300_with_copies.txt")

    # args_list = list([(root, features_dir, map_pe_dir, seed_index, fov, max_cloud) for root in roots])
    args_list = list([(root, data_config) for root in roots])
    num_processes = 64
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root, args_list):
            pbar.update()
            
    


def process_root(args):
    root, data_config = args

    # feature_path = f'{data_config['data_dir']}{data_config['features']['features_dir']}{root}.hdf5'
    segclr_path = f'{data_config['data_dir']}{data_config['segclr']['segclr_dir']}{root}.hdf5'

    try: 
        success_file_path = f"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/segclr_use_success/{root}"
        if os.path.exists(success_file_path):
            return
        with h5py.File(segclr_path, 'r+') as segclr_f:
            segclr_emb = segclr_f['segclr'][:]
            segclr_len = len(segclr_emb)
            # print("len", segclr_len)
            use = np.ones(segclr_len)
            
            all_zeros = np.all(segclr_emb == 0, axis=1)
            # print("all zeros", all_zeros)
            zero_indices = np.where(all_zeros)[0]
            # print("zero indices", zero_indices)
            if len(zero_indices) > 0:
                use[zero_indices] = 0
            # print("use", use)
            segclr_f.create_dataset('has_emb', data=use)
            with open(success_file_path, 'w') as file:
                pass

            # return node_count, error_count, conf_count, match_count, error_match_count, conf_smaller_than_cloud
    except Exception as e:
        print("root: ", root, "error: ", e)
        # return root
        return
        # return 0, 0, 0, 0, 0, 0
                
if __name__ == "__main__":

    test()
    