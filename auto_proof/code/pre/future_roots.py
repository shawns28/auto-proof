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
import glob
import dataset
from cloudvolume import CloudVolume
import multiprocessing
import time
import matplotlib.pyplot as plt
import argparse

'''
TODO: Fill in and remind that seg path is set in the private method
'''
def get_roots_at_mat(config, success_future_dir):
    data_directory = config['data']['data_dir']
    features_directory = config['data']['features_dir']
    root_paths = glob.glob(f'{features_directory}/*')

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

    root_paths = data_utils.get_roots_chunk(config, root_paths, chunk_num=chunk_num, num_chunks=num_chunks)

    num_processes= config['loader']['num_workers']

    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(root_paths)) as pbar:
        for root in pool.imap_unordered(__save_root943__, root_paths):
            with open(f'{success_future_dir}{root}', 'w') as f:
                pass
            pbar.update()

'''
TODO: Fill out
'''
def __save_root943__(root_path):
    # NOTE: Set the segmentation path here since we can't pass multiple things as params
    # seg_path = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1"
    seg_path = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m943/"
    cv_seg = CloudVolume(seg_path, use_https=True)
    root = root_path[-23:-15]
    resolution = np.array([[8, 8, 40]])
    with h5py.File(root_path, 'r+') as f:
        vertices = f['vertices'][:]
        vertices = vertices / resolution
        input_vertices = [(vertices[i][0].item(), vertices[i][1].item(), vertices[i][2].item()) for i in range(len(vertices))]
        num_tries = 3
        for i in range(num_tries):   
            try:
                root_943_dict = cv_seg.scattered_points(input_vertices)
                break
            except Exception as e:
                print("Failed to get scattered points for root", root, "error", e)
                if i < num_tries:
                    continue
                    
        root_943_arr = np.array([root_943_dict[input_vertices[i]] for i in range(len(vertices))])
        f.create_dataset('root_943', data=root_943_arr)
        return root

def save_post_roots(success_dir, post_path):
    files = glob.glob(f'{success_dir}*')
    print(files[0])
    file = files[0][-18:]
    print(file)
    roots = [files[i][-18:] for i in range(len(files))]
    data_utils.save_txt(post_path, roots)
    # shutil.rmtree(success_dir)

if __name__ == "__main__":
    config = data_utils.get_config()
    config['data']['is_proofread'] = False
    success_future_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/success_future_roots/"
    get_roots_at_mat(config, success_future_dir)

    post_future_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/post_future_roots.txt"
    save_post_roots(success_future_dir, post_future_path)