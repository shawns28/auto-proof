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
import multiprocessing
import time
import matplotlib.pyplot as plt
import graph_tool.all as gt

def __feature__(root_path):
    with h5py.File(root_path, 'r') as f:
        root_943s = f['root_943'][:]
        edges = f['edges'][:]
    g = gt.Graph(edges, directed=False)

    labels = get_labels(g, root_943s)

    confidences = get_confidences(root_943s)

    pos_enc = get_positional_encodings(g)

def parallel():
    roots = glob.glob('../../data/successful_root_943/*')
    root_paths = [f'../../data/features/{root[-18:]}_1000.h5py' for root in roots]

    num_processes = 1

    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(root_paths)) as pbar:
        for _ in pool.imap_unordered(__feature__, root_paths):
            pbar.update()

def main():
    # start_time = time.time()
    # data_directory = '../../data'
    # features_directory = f'{data_directory}/features'
    # root_paths = glob.glob(f'{features_directory}/*')
    # roots = glob.glob(f'{data_directory}/successful_root_943/*')
    # 864691135778235581 - 921 size
    # root_path = "../../data/features/864691135778235581_1000.h5py"
    # 864691135815797071 - 1000 size
    # root_path = "../../data/features/864691135815797071_1000.h5py"
    # 864691135937424949 - 7 size
    # root_path = "../../data/features/864691135937424949_1000.h5py"
    roots = glob.glob('../../data/successful_root_943/*')
    root_paths = [f'../../data/features/{root[-18:]}_1000.h5py' for root in roots]
    # root_paths = ['../../data/864691135815797071_1000.h5py']

    chunk_num = int(args[0])
    num_chunks = 24
    print("chunk number is:", chunk_num)
    print("num_chunks is:", num_chunks)
    chunk_size = len(root_paths) // num_chunks
    start_index = (chunk_num - 1) * chunk_size
    end_index = start_index + chunk_size + 1
    if chunk_num == num_chunks:
        root_paths = root_paths[start_index:]
    else:
        root_paths = root_paths[start_index:end_index]

    with tqdm(total=len(root_paths)) as pbar:
        for root_path in root_paths:
            with h5py.File(root_path, 'r+') as f:
                root_943s = f['root_943'][:]
                edges = f['edges'][:]
                # print("root len", len(root_943s))
                g = gt.Graph(edges, directed=False)
                # start_labels = time.time()
                labels = get_labels(g, root_943s)
                f.create_dataset('label', data=labels)
                # end_labels = time.time()
                # print("labels time", end_labels - start_labels)

                # test_set_time(root_943s)
                # start_confidences = time.time()
                confidences = get_confidences(root_943s)
                f.create_dataset('confidence', data=confidences)
                # print("confidence mismatch", np.where(confidences != True)[0])
                # end_confidences = time.time()
                # print("confdicences time", end_confidences - start_confidences)

                # start_posenc = time.time()
                pos_enc = get_positional_encodings(g)
                f.create_dataset('pos_enc', data=pos_enc)
                # end_posenc = time.time()
                # print("pos enc time", end_posenc - start_posenc)

                # print("total time", end_posenc - start_time)
                with open(f'../../data/successful_labels/{root_path[-28:-10]}', 'w') as f:
                    pass
                pbar.update()
    
            # with h5py.File(root_path, 'r') as f:
            #     labels = f['label'][:]
            #     c = f['confidence'][:]
            #     pos_enc = f['pos_enc'][:]
            #     print("labels", labels)
            #     print("conf", c)
            #     print("pos enc", pos_enc)

def get_labels(g, root_943s):
    labels = np.full(len(root_943s), True)
    for v in g.iter_vertices():
        root_943 = root_943s[v]
        # print(v)
        for e in g.iter_out_edges(v):
            # print(root_943s[e[0]], root_943s[e[1]])
            if root_943s[e[1]] != root_943:
                labels[v] = False
                break
    # print("labels mismatch", np.where(labels != True)[0])
    # print("943 mismatch", np.where(root_943s != 864691135279409057)[0])
    return labels

def get_confidences(root_943s):
    proofread_roots = data_utils.load_txt('../../data/proofread_943.txt')
    return np.isin(root_943s, proofread_roots)

def get_positional_encodings(g, pos_enc_dim=32):
    '''
    Adapted from https://github.com/marissaweis/ssl_neuron/blob/main/ssl_neuron/utils.py
    which was adapted from 
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/ef8bd8c7d2c87948bc1bdd44099a52036e715cd0/data/molecules.py#L147-L168.
    '''
    norm_lapl = gt.laplacian(g, norm=True).toarray()
    # print(norm_lapl)
    eig_val, eig_vec = np.linalg.eigh(norm_lapl)
    # print("eig val", eig_val)
    # print("eig_vec", eig_vec)
    # print("eig_vec shape", eig_vec.shape)
    eig_vec = np.flip(eig_vec, axis=[1])
    pos_enc = eig_vec[:, 1:pos_enc_dim + 1]

    if pos_enc.shape[1] < pos_enc_dim:
        pos_enc = np.concatenate([pos_enc, np.zeros((pos_enc.shape[0], pos_enc_dim - pos_enc.shape[1]))], axis=1)
    # print("pos enc", pos_enc)
    # print("pos enc shape", pos_enc.shape)
    return pos_enc

def test_set_time(root_943s):
    proofread_roots = data_utils.load_txt('../../data/proofread_943.txt')
    proofread_set = set(proofread_roots)
    set_time = time.time()
    res = [root in proofread_set for root in root_943s]
    set_end_time = time.time()
    res = np.isin(root_943s, proofread_roots)
    isin_end_time = time.time()

    print("set time", set_end_time - set_time)
    print("isin time", isin_end_time - set_end_time)

if __name__ == "__main__":
    main()
    # parallel()