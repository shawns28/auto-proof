from auto_proof.code.pre import data_utils

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import json
import sys
import time
import argparse
import torch
import multiprocessing
import glob

def rename_files(folder_dir, dir_len):
    with tqdm(total=dir_len) as pbar:
        for filename in os.listdir(folder_dir):
            source = os.path.join(folder_dir, filename)
            if os.path.isfile(source):  # Ensure it's a file
                destination = os.path.join(folder_dir, conversion_function(filename))
                # print("destination", destination)
                os.rename(source, destination)
                pbar.update()

def conversion_function(filename):
    root_and_metadata = filename[:-5]
    # print(root_and_metadata)
    root_and_metadata += '_000'
    return root_and_metadata + filename[-5:]

def convert_roots_in_txt(root_path):
    roots = data_utils.load_txt(root_path)
    new_roots = []
    for i in range(len(roots)):
        new_roots.append(str(roots[i]) + '_000')
    new_roots = np.array(new_roots)
    new_root_path = root_path[:-4] + '_conv' + root_path[-4:]
    data_utils.save_txt(new_root_path, new_roots)

# folder_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features_conf/"
# dir_len = 500000
# rename_files(folder_dir, dir_len)

root_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/val_roots_45333_69.txt"
convert_roots_in_txt(root_path)