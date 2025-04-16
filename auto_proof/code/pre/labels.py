from auto_proof.code.pre import data_utils

import numpy as np
import time
import h5py
import networkx as nx


def create_labels(root, root_at_arr, feature_path, proofread_roots_path):
    '''Creates labels, confidences and positional encodings
    
    TODO: Fill in 
    Assumes root_to_arr has roots that don't have identifiers
    '''
    with h5py.File(feature_path, 'r') as f:
        edges = f['edges'][:]
        g = nx.Graph()
        g.add_nodes_from(range(len(root_at_arr)))
        g.add_edges_from(edges)
        labels = get_labels(g, root_at_arr)
        confidences = get_confidences(root_at_arr, labels, proofread_roots_path)
        return labels, confidences

def get_labels(g, root_at_arr):
    labels = np.full(len(root_at_arr), False)
    for v in g.nodes():
        root_at = root_at_arr[v]
        for e in g.edges(v):
            if root_at_arr[e[1]] != root_at:
                labels[v] = True
                break
    return labels

def get_confidences(root_at_arr, labels, proofread_roots_path):
    proofread_roots = data_utils.load_txt(proofread_roots_path)
    conf = np.isin(root_at_arr, proofread_roots)
    # Confidence at errors should be True
    conf[labels == True] = True
    return conf