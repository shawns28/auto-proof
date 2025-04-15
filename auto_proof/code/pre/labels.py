from auto_proof.code.pre import data_utils

import numpy as np
import time
import h5py
import networkx as nx


def create_labels(root, root_at_arr, edges, data_config):
    '''Creates labels, confidences and positional encodings
    
    TODO: Fill in 
    Assumes root_to_arr has roots that don't have identifiers
    '''
    feature_path = f'{data_config['data_dir']}{data_config['features']['features_dir']}{root}.hdf5'
    with h5py.File(feature_path, 'r') as f:
        edges = f['edges'][:]
        g = nx.Graph()
        g.add_nodes_from(range(len(root_at_arr)))
        g.add_edges_from(edges)
        labels = get_labels(g, root_at_arr)
        confidences = get_confidences(data_config, root_at_arr, labels)
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

def get_confidences(data_config, root_at_arr, labels):
    mat_version1 = data_config['proofread']['mat_versions'][0]
    mat_version2 = data_config['proofread']['mat_versions'][1]
    proofread_roots = data_utils.load_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version1}_{mat_version2}.txt')
    conf = np.isin(root_at_arr, proofread_roots)
    # Confidence at errors should be True
    conf[labels == True] = True
    return conf