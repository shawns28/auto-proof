from auto_proof.code.pre import data_utils

# from caveclient import CAVEclient
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
import argparse
import shutil

'''
Creates labels, confidences and positional encodings
'''
def create_labels(config, success_labels_dir):
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

    roots = data_utils.load_txt(config['data']['root_path'])
    roots = data_utils.get_roots_chunk(config, roots, chunk_num=chunk_num, num_chunks=num_chunks)

    if not os.path.isdir(success_labels_dir):
        os.makedirs(success_labels_dir)

    with tqdm(total=len(roots)) as pbar:
        for root in roots:            
            success_path = f'{success_labels_dir}{root}'
            # Skip already processed roots
            if os.path.exists(success_path):
                pbar.update()
                continue
            
            root_path = f'{config['data']['features_dir']}{root}.hdf5'
            with h5py.File(root_path, 'r+') as f:
                root_943s = f['root_943'][:]
                edges = f['edges'][:]
    
                g = gt.Graph(edges, directed=False)

                labels = get_labels(g, root_943s)
                f.create_dataset('label', data=labels)

                confidences = get_confidences(config, root_943s, labels)
                f.create_dataset('confidence', data=confidences)

                pos_enc = get_positional_encodings(config, g)
                f.create_dataset('pos_enc', data=pos_enc)

                with open(f'{success_labels_dir}{root}', 'w') as f:
                    pass
                pbar.update()

def get_labels(g, root_943s):
    labels = np.full(len(root_943s), True)
    for v in g.iter_vertices():
        root_943 = root_943s[v]
        for e in g.iter_out_edges(v):
            if root_943s[e[1]] != root_943:
                labels[v] = False
                break
    return labels

def get_confidences(config, root_943s, labels):
    proofread_roots_at_mat = data_utils.load_txt(config['data']['proofread_at_mat_path'])
    conf = np.isin(root_943s, proofread_roots_at_mat)
    # Confidence at errors should be 1
    conf[labels == False] = True
    return conf

def get_positional_encodings(config, g):
    '''
    Adapted from https://github.com/marissaweis/ssl_neuron/blob/main/ssl_neuron/utils.py
    which was adapted from 
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/ef8bd8c7d2c87948bc1bdd44099a52036e715cd0/data/molecules.py#L147-L168.
    '''
    pos_enc_dim = config['data']['pos_enc_dim']
    norm_lapl = gt.laplacian(g, norm=True).toarray()
    eig_val, eig_vec = np.linalg.eigh(norm_lapl)
    eig_vec = np.flip(eig_vec, axis=[1])
    pos_enc = eig_vec[:, 1:pos_enc_dim + 1]

    if pos_enc.shape[1] < pos_enc_dim:
        pos_enc = np.concatenate([pos_enc, np.zeros((pos_enc.shape[0], pos_enc_dim - pos_enc.shape[1]))], axis=1)
    return pos_enc

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
    config['data']['is_proofread'] = True
    config['data']['features_dir'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/proofread_features/"
    config['data']['root_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/post_skel_proofread_roots.txt"
    success_labels_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/success_labels_proofread/"
    create_labels(config, success_labels_dir)

    post_labels_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/post_labels_proofread_roots.txt"
    save_post_roots(success_labels_dir, post_labels_path)