import data_utils
from caveclient import CAVEclient
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import json
import fastremap
import sys
import time

def main(args):
    config, _, _, _, data_directory = data_utils.initialize()
    # data_directory = '../../testing_data'
    features_directory = f'{data_directory}/features_test'
    # features_directory = f'{data_directory}/features'
    if not os.path.isdir(features_directory):
        os.makedirs(features_directory)

    chunk_num = int(args[0])
    num_chunks = 24

    root_ids = data_utils.get_roots_chunk(chunk_num=chunk_num, num_chunks=num_chunks)
    
    cutoff = config["data"]["cutoff"]

    root_ids = ["864691136272969918"]
    root_to_num_vertices = {}
    root_to_num_init_vertices = {}
    for root_id in root_ids:
        root_dir = f'{features_directory}/{root_id}'
        skel_hf_path = f'{root_dir}/{root_id}_{cutoff}.h5py'
        with h5py.File(skel_hf_path, 'r') as f:
            num_vertices = f['num_vertices']
            num_initial_vertices = f['num_initial_vertices']
        root_to_num_vertices[root_id] = num_vertices
        root_to_num_init_vertices[root_id] = num_initial_vertices

    

if __name__ == "__main__":
    main(sys.argv[1:])