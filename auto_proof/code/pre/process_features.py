from auto_proof.code.pre import data_utils
from auto_proof.code.pre.process_raw_edits import process_raw_edits
from auto_proof.code.pre.create_proofread_943_txt import convert_proofread_csv_to_txt

import os
import numpy as np
from tqdm import tqdm
import graph_tool.all as gt
import h5py
import time
import argparse
import multiprocessing

def main(data_config):
    """
        TODO: Fill in
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chunk_num", help="chunk num")
    parser.add_argument("-n", "--num_processes", help="num processes")
    args = parser.parse_args()
    if args.num_processes:
        data_config['multiprocessing']['num_processes'] = int(args.num_processes)
    chunk_num = 1
    num_chunks = config['multiprocessing']['num_chunks']
    if args.chunk_num:
        chunk_num = int(args.chunk_num)
    else: # No chunking
        num_chunks = 1

    roots = data_utils.load_txt(data_config['paths']['root_path'])
    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)

    num_processes = data_config['multiprocessing']['num_processes']
    client, _, _ = data_utils.create_client(data_config)
    args_list = list([(root, data_config, client) for root in roots])
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root, args_list):
            pbar.update()

    # Save all the roots that were successful


def process_root_features(data):
    """
       TODO: Fill in
    """
    root, data_config, client = data
    # Skeletonize a
    

    # map pe  r+

    # dist r+

    pass


if __name__ == "__main__":
    data_config = data_utils.get_data_config()
    main(data_config)