from auto_proof.code.pre import data_utils

import networkx as nx
import numpy as np
from tqdm import tqdm
import multiprocessing
import glob
import os
import h5py
import argparse

def create_dist(root, edges, labels):
    """
    TODO: Fill in
    """
    g = nx.Graph()
    g.add_nodes_from(range(len(labels)))
    g.add_edges_from(edges)
    distances = get_distances(g, labels)
    return distances

def get_distances(g, labels):
    error_vertices = [v for v in range(len(labels)) if labels[v] == True]

    max_int32 = np.iinfo(np.int32).max
    distances = np.full(len(labels), max_int32, dtype=np.int32)

    if len(error_vertices) == 0:
        return distances
    
    for v in error_vertices:
        shortest_paths = nx.shortest_path_length(g, source=v)
        for u, dist in shortest_paths.items():
            if dist < distances[u]:
                distances[u] = dist
    return distances