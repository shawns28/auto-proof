from auto_proof.code.pre import data_utils

from caveclient import CAVEclient
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import json
import sys
import time
import os
import shutil
import glob
import multiprocessing


'''
TODO: Fill in all of the methods
'''
def convert_proofread_csv_to_txt(config, mat_version):
    data_dir = config['data']['data_dir']
    proofread_csv = f'{data_dir}proofread_{mat_version}.csv'
    df = pd.read_csv(proofread_csv)
    filtered_df = df[df['status_axon'] != 'non']
    root_ids = filtered_df['root_id']
    root_ids_array = np.array(root_ids)
    data_utils.save_txt(f'{data_dir}root_ids/proofread_{mat_version}.txt', root_ids_array)

'''
TODO: No longer works since I took away proofread path
'''
def proofread_future_roots(config):
    proofread_roots = data_utils.load_txt(config['data']['proofread_path'])
    for root in proofread_roots:
        skel_hf_path = f'{config['data']['proofread_features_dir']}{root}.hdf5'
        with h5py.File(skel_hf_path, 'a') as skel_hf:
            if 'root_943' not in skel_hf:
                num_vertices = skel_hf['num_vertices'][()]
                arr = np.full(num_vertices, root)  
                skel_hf.create_dataset('root_943', data=arr)

def combine_roots(previous_root_path, new_root_path, combined_root_path):
    previous_roots = data_utils.load_txt(previous_root_path)
    new_roots = data_utils.load_txt(new_root_path)
    combined_roots = np.concatenate((previous_roots, new_roots))
    print(combined_roots[0], " ", combined_roots[-1])
    data_utils.save_txt(combined_root_path, combined_roots)

'''
Adds new features into previous features dir
'''
def combine_feat_dir(previous_feat_dir, new_feat_dir):
    source = new_feat_dir
    destination = previous_feat_dir
    
    # gather all files
    allfiles = os.listdir(source)
    
    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        shutil.move(src_path, dst_path)

# NOTE: This is bad style but I'm doing for the multiprocessing
config = data_utils.get_config()
PROOFREAD_ROOTS = data_utils.load_txt(config['data']['proofread_at_mat_path'])
def add_proofread_feature(config):
    root_paths = glob.glob(f'{config['data']['features_dir']}*')
    roots = data_utils.load_txt(config['data']['root_path'])
    proofread_roots = data_utils.load_txt(config['data']['proofread_at_mat_path'])
    num_processes = config['loader']['num_workers']

    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(root_paths)) as pbar:
        for _ in pool.imap_unordered(__process_add_proofread_feature__, root_paths):
            pbar.update()

def __process_add_proofread_feature__(root_path):
    with h5py.File(root_path, 'r+') as f:
        if 'is_proofread' not in f:
            root = f['root_id'][()]
            if root in PROOFREAD_ROOTS:
                f.create_dataset('is_proofread', data=True)
            else:
                f.create_dataset('is_proofread', data=False)


if __name__ == "__main__":
    mat_version = 943
    config = data_utils.get_config()
    # convert_proofread_csv_to_txt(config, mat_version)
    # proofread_future_roots(config)
    # previous_root_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/post_label_roots_459972.txt"
    # new_root_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/post_labels_proofread_roots.txt"
    # combined_root_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/all_roots.txt"
    # combine_roots(previous_root_path, new_root_path, combined_root_path)
    # previous_feat_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features_conf/"
    # new_feat_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/proofread_features/"
    # combine_feat_dir(previous_feat_dir, new_feat_dir)

    add_proofread_feature(config)
        
