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
import sys
from auto_proof.code.connectomics.reader import EmbeddingReader
from auto_proof.code.connectomics.sharding import md5_shard
import gcsfs
from scipy.spatial import cKDTree

def run_segclr():
    data_config = data_utils.get_data_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chunk_num", help="chunk num")
    parser.add_argument("-n", "--num_processes", help="num processes")
    args = parser.parse_args()
    if args.num_processes:
        data_config['multiprocessing']['num_processes'] = int(args.num_processes)
    chunk_num = 1
    num_chunks = data_config['multiprocessing']['num_chunks']
    if args.chunk_num:
        chunk_num = int(args.chunk_num)
    else: # No chunking
        num_chunks = 1

    # roots = data_utils.load_txt(config['data']['root_path'])
    # roots = ['864691135591041291_000'] # proofread at 943
    roots = ['864691135463333789_000']
    # roots = ['864691135937424949_000'] # size 7
    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)

    num_processes = data_config['multiprocessing']['num_processes']
    features_directory = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features_conf/"
    mat = 943

    # NOTE: This is for 943 specifically
    filesystem = gcsfs.GCSFileSystem(token='anon')
    url = 'gs://iarpa_microns/minnie/minnie65/embeddings_m943/segclr_nm_coord_public_offset_csvzips/'

    num_shards = 50_000
    bytewidth = 8
    embedding_reader = EmbeddingReader(filesystem, url, sharder, num_shards, bytewidth)

    # filesystem = gcsfs.GCSFileSystem(token='anon')
    # url = 'gs://iarpa_microns/minnie/minnie65/embeddings_m1300/segclr_nm_coord_public_offset_csvzips/'

    # num_shards = 50_000
    # bytewidth = 8
    # embedding_reader = EmbeddingReader(filesystem, url, sharder, num_shards, bytewidth)

    args_list = list([(root, features_directory, mat, embedding_reader) for root in roots])

    # with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
    #     for _ in pool.imap_unordered(segclr_one_root, args_list):
    #         pbar.update() 
    
    segclr(args_list[0])           

def segclr(data):
    root, features_directory, mat, embedding_reader = data
    root_feat_path = f'{features_directory}{root}.hdf5'
    with h5py.File(root_feat_path, 'r') as f:
        vertices = f['vertices'][:]
        print("num vertice", vertices.shape)
        roots_at_mat = f[f'root_{mat}'][:]

        # Testing max distance of vertices
        max_dist = 0
        vert_tree = cKDTree(vertices)
        for vertice in vertices:
            dd, ii = vert_tree.query(vertice, k=2)
            max_dist = max(max_dist, dd[1])
        print("max dist in vertices", max_dist)

        roots_at_mat_to_indices = create_root_at_mat_dict(roots_at_mat)
        # print("roots_at_mat_to_indices", roots_at_mat_to_indices)

        # Create a map from root_943 to list of vertice indices
        # This way we can process each root_943 at a time and get rid of embeddings after

        result = np.zeros((len(vertices), 128))
        for root in roots_at_mat_to_indices:
            embs = embedding_reader[root]

            # Convert to using the actual indices
            emb_vals = []
            coords = []
            for coord_key, emb_val in embs.items():
                emb_vals.append(emb_val)
                coords.append(coord_key)
            emb_vals = np.array(emb_vals)

            print("emb vals shape", emb_vals.shape)

            # radius = 2000 # 2 um
            radius = 3000
            # radius = 300
            # radius = 500
            tree = cKDTree(coords)

            for index in roots_at_mat_to_indices[root]:
                vertice = vertices[index]
                # print("coords at index", vertices[index], index)
                # query with a max distance
                # dd, ii = tree.query(vertice, k=7)
                # print("dd", dd)
                # print("ii", ii)
                # for i in ii:
                #     print("closest vertices", coords[i])

                q_indices = tree.query_ball_point(vertice, radius)
                # print("indices", q_indices)
                for i in q_indices:
                    print("closest vertices", coords[i])
                if len(q_indices) == 0:
                    print("Nothing within radius of:", radius)

                result[index] = np.mean([emb_vals[q_ind] for q_ind in q_indices], axis=0)
        print("result shape", result.shape)

def create_root_at_mat_dict(roots_at_mat):
    roots_at_mat_to_indices = {}
    for i in range(len(roots_at_mat)):
        root = int(roots_at_mat[i])
        if root not in roots_at_mat_to_indices:
            roots_at_mat_to_indices[root] = []
        roots_at_mat_to_indices[root].append(i)
    return roots_at_mat_to_indices

def sharder(segment_id: int, num_shards, bytewidth) -> int:
    return md5_shard(segment_id, num_shards=num_shards, bytewidth=bytewidth)

if __name__ == "__main__":    
    run_segclr()