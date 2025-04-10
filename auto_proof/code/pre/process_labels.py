from auto_proof.code.pre import data_utils
from auto_proof.code.pre.process_raw_edits import process_raw_edits
from auto_proof.code.pre.create_proofread_943_txt import convert_proofread_csv_to_txt

import os
import numpy as np
from tqdm import tqdm
import h5py
import time
import argparse
import multiprocessing
from cloudvolume import CloudVolume


def main(data_config):
    """
        TODO: Fill in
    """
    data_config, chunk_num, num_chunks, num_processes = data_utils.get_num_chunk_and_processes(data_config)

    roots = data_utils.load_txt()
    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)

    labels_dir = data_config['features']['labels_dir']
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    roots_at_latest_dir = data_config['features']['roots_at_latest_dir']
    if not os.path.exists(roots_at_latest_dir):
        os.makedirs(roots_at_latest_dir)

    latest_version = data_config['labels']['latest_mat_version']
    seg_path = data_config['segmentation'][f'precomputed_{latest_version}']
    cv_seg = CloudVolume(seg_path, use_https=True)

    root_paths = 

    args_list = list([(root, data_config, client) for root in roots])
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root, args_list):
            pbar.update()

    # Save all the roots that were successful

def process_root(data):
    """
        TODO: Fill in
    """
    root, data_config, client = data
    # If flag for process root features

    # If process root at

    # If flag for labels

    # If flag for segclr


def process_root_features(config: dict, root: int, rep_coord: list):
    """
       TODO: Fill in
    """
    # Skeletonize a

    # map pe  r+

    # dist r+

    pass

def process_root_at(config: dict, root: int):
    """
        TODO: Fill in
    """
    pass

def process_root_labels(config: dict, root: int):
    """
        TODO: Fill in
    """
    pass

def process_segclr(config: dict, root: int):
    """
        TODO: Fill in
    """
    pass


if __name__ == "__main__":
    config = data_utils.get_config()
    data_config = data_utils.get_data_config()



    main()