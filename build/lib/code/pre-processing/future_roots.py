import data_utils
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

from .. import dataset

config, _, _, _, data_directory = data_utils.initialize()
# features_directory = f'{data_directory}/features_test'
# features_directory = f'{data_directory}/features'
features_directory = f'{data_directory}/features_converted_chunked_1000'
figures_path = f'{data_directory}/figures'
def main():
    # dataset = 
    with tqdm(total=len(roots)) as pbar:
        for data in loader:
            root, vertices, edges, rank_0, rank_1, rank_2, num_vertices, num_initial_vertices, compartment, radius = data
        pbar.update()

def test_h5_add():
    hf = h5py.File(f'{data_directory}/num_vertices.hdf5', 'r+')
    root = '111111111111111111'
    num_vertices = [[11]]
    hf.create_dataset(root, data=num_vertices)
    hf.close()
    print("done adding dataset")

    # root = '864691135815797071'
    rf = h5py.File(f'{data_directory}/num_vertices.hdf5', 'r')
    print(rf[root][()])
    num_vertices = rf[root][()]
    print(num_vertices[0][0].dtype)

def get_roots_467853():
    files = glob.glob(f'{features_directory}/*')
    print(files[0])
    file = files[0][20:38]
    print(file)
    roots = [files[i][20:38] for i in range(len(files))]
    data_utils.save_txt(f'{data_directory}/roots_467853.txt', roots)


if __name__ == "__main__":
    # test_h5_add()
    # main()
    print("hi")
    