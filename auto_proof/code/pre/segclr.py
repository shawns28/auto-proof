from auto_proof.code.pre import data_utils
from auto_proof.code.visualize import visualize_segclr


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

def get_segclr_emb(root, feature_path, root_at_arr, embedding_reader, emb_dim):
    with h5py.File(feature_path, 'r') as feat_f:
        vertices = feat_f['vertices'][:]
        edges = feat_f['edges'][:]
        print("num vertice", vertices.shape)

        # Testing max distance of vertices
        max_dist = 0
        vert_tree = cKDTree(vertices)
        for vertice in vertices:
            dd, ii = vert_tree.query(vertice, k=2)
            max_dist = max(max_dist, dd[1])
        print("max dist in vertices", max_dist)

        roots_at_to_indices = create_root_at_dict(root_at_arr)

        # Create a map from root_943 to list of vertice indices
        # This way we can process each root_943 at a time and get rid of embeddings after

        result = np.zeros((len(vertices), emb_dim))
        for root in roots_at_to_indices:
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
            # radius = 3000
            # radius = 300
            radius = 500
            tree = cKDTree(coords)

            for index in roots_at_to_indices[root]:
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
                # for i in q_indices:
                #     print("closest vertices", coords[i])
                if len(q_indices) == 0:
                    print("Nothing within radius of:", radius)

                result[index] = np.mean([emb_vals[q_ind][:emb_dim] for q_ind in q_indices], axis=0)
                #TODO: Need to get an array that has the vertices of the segclr nodes, might need to visualize a subset if its not able to visualize all the points
        print("result shape", result.shape)
        # path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/segclr_test/{root}_segclr.html'
        # visualize_segclr(vertices, edges, res, path)
        return True, None, result

def create_root_at_dict(roots_at):
    roots_at_to_indices = {}
    for i in range(len(roots_at)):
        root = int(roots_at[i])
        if root not in roots_at_to_indices:
            roots_at_to_indices[root] = []
        roots_at_to_indices[root].append(i)
    return roots_at_to_indices

def sharder(segment_id: int, num_shards, bytewidth) -> int:
    return md5_shard(segment_id, num_shards=num_shards, bytewidth=bytewidth)

if __name__ == "__main__":    
    run_segclr()