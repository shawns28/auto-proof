from auto_proof.code.pre import data_utils
from auto_proof.code.visualize import visualize_segclr

import networkx as nx
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

def get_segclr_emb(root, feature_path, root_at_arr, embedding_reader, emb_dim, visualize_radius, small_radius, large_radius):
    try: 
        with h5py.File(feature_path, 'r') as feat_f:
            vertices = feat_f['vertices'][:]
            edges = feat_f['edges'][:]
            # print("num vertice", vertices.shape)
            g = nx.Graph()
            g.add_nodes_from(range(len(vertices)))
            g.add_edges_from(edges)

            roots_at_to_indices = create_root_at_dict(root_at_arr)

            # Create a map from root_943 to list of vertice indices
            # This way we can process each root_943 at a time and get rid of embeddings after

            original_root = root
            result = np.zeros((len(vertices), emb_dim))
            has_emb = np.ones(len(vertices))
            if visualize_radius:
                too_far_small_radius_indices = np.zeros(len(vertices), dtype=bool)
                too_far_large_radius_indices = np.zeros(len(vertices), dtype=bool)
            for root in roots_at_to_indices:
                try:
                    embs = embedding_reader[root]
                except KeyError as e: # If is doesn't exist in segclr archive
                    continue
                except Exception as e:
                    return False, e, None, None
                # Convert to using the actual indices
                emb_vals = []
                coords = []
                for coord_key, emb_val in embs.items():
                    emb_vals.append(emb_val)
                    coords.append(coord_key)
                emb_vals = np.array(emb_vals)
                coords = np.array(coords)

                # print("emb vals shape", emb_vals.shape)

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

                    q_indices = tree.query_ball_point(vertice, small_radius)
                    if len(q_indices) == 0:
                        too_far_for_large = True
                        # print("Nothing within radius of:", small_radius)
                        degree_of_vertex = g.degree(index)
                        if degree_of_vertex > 4:
                            q_dist, q_indices = tree.query(vertice, k=10, distance_upper_bound=large_radius)
                            q_indices = q_indices[np.where(q_dist != float('inf'))]
                            if not len(q_indices) == 0:
                                result[index] = np.mean([emb_vals[q_ind][:emb_dim] for q_ind in q_indices], axis=0)
                                too_far_for_large = False
                        elif degree_of_vertex == 1:
                            q_d, q_ind = tree.query(vertice, k=1, distance_upper_bound=large_radius)
                            if not q_d == float('inf'):
                                result[index] = emb_vals[q_ind][:emb_dim]
                                too_far_for_large = False
                        if visualize_radius:
                            if too_far_for_large:
                                too_far_large_radius_indices[index] = True
                            else:
                                too_far_small_radius_indices[index] = True
                        # if its a soma then take average of everything around it so that model knows its a soma
                        # if its not a soma then take the closest one
                    else:
                        result[index] = np.mean([emb_vals[q_ind][:emb_dim] for q_ind in q_indices], axis=0)            
            all_zeros = np.all(result == 0, axis=1)
            zero_indices = np.where(all_zeros)[0]
            if len(zero_indices) > 0:
                has_emb[zero_indices] = 0
            # print("use", use)
            # print("result shape", result.shape)
            if visualize_radius:
                path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/segclr_test/{original_root}_{small_radius}_{large_radius}radius_segclr.html'
                visualize_segclr(vertices, edges, coords, too_far_small_radius_indices, too_far_large_radius_indices, path)
            return True, None, result, has_emb
    except Exception as e:
        return False, e, None, None

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