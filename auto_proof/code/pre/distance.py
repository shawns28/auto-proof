from auto_proof.code.pre import data_utils

import networkx as nx
import numpy as np
from tqdm import tqdm
import multiprocessing
import glob
import os
import h5py
import argparse
from typing import List, Tuple, Any

def create_dist(edges: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Computes the shortest distance from any 'error' vertex to all other vertices in a skeleton graph.

    Args:
        edges: A NumPy array of shape `(N, 2)` representing the edges of the
            skeleton graph. Each row `[u, v]` indicates an edge between vertex `u` and `v`.
        labels: A NumPy boolean array of shape `(M,)` where `M` is the number of vertices.
            `labels[i] == True` indicates that vertex `i` is an 'error' vertex.

    Returns:
        A NumPy array of shape `(M,)` containing the shortest distance from any
        'error' vertex to each vertex in the graph. If no 'error' vertices are
        present, all distances will be set to the maximum `int32` value.
    """
    g = nx.Graph()
    g.add_nodes_from(range(len(labels)))
    g.add_edges_from(edges)
    distances = get_distances(g, labels)
    return distances

def get_distances(g: nx.Graph, labels: np.ndarray) -> np.ndarray:
    """Calculates the minimum shortest path distance from any 'error' vertex to all other vertices.

    Args:
        g: A NetworkX graph representing the skeleton.
        labels: A NumPy boolean array where `labels[i] == True` indicates vertex `i` is an 'error' vertex.

    Returns:
        A NumPy array of the same length as `labels`, where each element `distances[i]`
        is the shortest path distance from vertex `i` to the nearest 'error' vertex.
        If no 'error' vertices are found, all distances are set to `np.iinfo(np.int32).max`.
    """
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