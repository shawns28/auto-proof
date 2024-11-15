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
import matplotlib.pyplot as plt
import glob

config, _, _, _, data_directory = data_utils.initialize()
# features_directory = f'{data_directory}/features_test'
features_directory = f'{data_directory}/features'
figures_path = f'{data_directory}/figures'
root_num_vertices_path = f'{data_directory}/root_to_num_vertices_467853.pkl'
root_num_init_vertices_path = f'{data_directory}/root_to_num_init_vertices_467853.pkl'

def save_dicts():
    root_ids = data_utils.get_roots()
    
    cutoff = config["data"]["cutoff"]

    # root_ids = [864691136272969918, 864691135776571232, 864691135474550843, 864691135445638290, 864691136899949422, 864691136175092486, 864691135937424949]
    # root_ids = [864691136272969918, 864691135776571232, 864691135474550843, 864691136899949422, 864691136175092486, 864691135937424949]

    root_to_num_vertices = {}
    root_to_num_init_vertices = {}
    for root_id in root_ids:
        root_dir = f'{features_directory}/{root_id}'
        skel_hf_path = f'{root_dir}/{root_id}_{cutoff}.h5py'
        with h5py.File(skel_hf_path, 'r') as f:
            num_vertices = f['num_vertices'][()]
            num_initial_vertices = f['num_initial_vertices'][()]
        root_to_num_vertices[root_id] = num_vertices
        root_to_num_init_vertices[root_id] = num_initial_vertices
    data_utils.save_pickle_dict(root_num_vertices_path, root_to_num_vertices)
    data_utils.save_pickle_dict(root_num_init_vertices_path, root_to_num_init_vertices)

def save_dicts_2():
    root_to_num_vertices = {}
    root_to_num_init_vertices = {}
    files = glob.glob(f'{features_directory}/*')
    no_file_count = 0
    with tqdm(total=len(files)) as pbar:
        for file in files:
            # substring = file[20:]
            # path = f'{features_directory}/{substring}/{substring}_1000.h5py'
            path = file
            if os.path.exists(path):
                with h5py.File(path, 'r') as f:
                    num_vertices = f['num_vertices'][()]
                    num_initial_vertices = f['num_initial_vertices'][()]
                root_to_num_vertices[file] = num_vertices
                root_to_num_init_vertices[file] = num_initial_vertices
            else:
                no_file_count += 1
            pbar.update()
    print("no file count", no_file_count)
    data_utils.save_pickle_dict(root_num_vertices_path, root_to_num_vertices)
    data_utils.save_pickle_dict(root_num_init_vertices_path, root_to_num_init_vertices)

def main():   
    root_to_num_init_vertices = data_utils.load_pickle_dict(root_num_init_vertices_path)
    data = np.array(list(root_to_num_init_vertices.values()))
    plot_data(data, 'Distribution of Initial Vertice Counts', figures_path)
    root_to_num_vertices = data_utils.load_pickle_dict(root_num_vertices_path)
    data = np.array(list(root_to_num_vertices.values()))
    plot_data(data, f'Distribution of Vertice Counts Post Cutoff {config['data']['cutoff']}', figures_path)

def plot_data(data, title, path):
    # bins = [0, 25, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 100000]
    # Logarithmic bins
    bins = np.logspace(np.log10(min(data)), np.log10(max(data)), 10)
    # quantiles = np.quantile(data, np.linspace(0, 1, 10))
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xscale('log')
    plt.xlabel('Vertices')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(f'{path}/{title}_467853_log')
    plt.close()

def get_directory_size():
    files = glob.glob(f'{features_directory}/*')
    print("Number of files", len(files))
    print("first file", files[0])
    print("stats of first file", os.stat(files[0]))
    size_array = np.zeros(len(files))
    with tqdm(total=len(files)) as pbar:
        for i, file in enumerate(files):
            file_stats = os.stat(file)
            file_size_bytes = file_stats.st_size
            size_array[i] = file_size_bytes / 1000
            pbar.update()
    print("total size in kb: ", size_array.sum())
    plot_data(size_array, "Distribution of size of file in kb", figures_path)
    print("total size of directory", os.stat(features_directory))

def get_unskel():
    files = glob.glob(f'{features_directory}/*')
    roots = np.array([files[i][20:38] for i in range(len(files))])
    all_roots = data_utils.get_roots()
    diff2 = np.setdiff1d(all_roots, roots)
    print(diff2)
    print(len(diff2))
    data_utils.save_txt(f'{data_directory}/unprocessed_roots.txt', diff2)

if __name__ == "__main__":
    # get_unskel()
    # get_directory_size()
    # save_dicts_2()
    main()
