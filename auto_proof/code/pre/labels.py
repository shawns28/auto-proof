from auto_proof.code.pre import data_utils

import numpy as np
import time
import h5py
import networkx as nx


def create_labels(root, root_at_arr, ignore_edge_ccs, edges, proofread_roots):
    '''Creates labels, confidences and positional encodings
    
    TODO: Fill in 
    Assumes root_to_arr has roots that don't have identifiers
    '''
    
    g = nx.Graph()
    g.add_nodes_from(range(len(root_at_arr)))
    g.add_edges_from(edges)
    labels, changed_ccs = get_labels(g, root_at_arr, ignore_edge_ccs)
    confidences = get_confidences(root_at_arr, labels, changed_ccs, proofread_roots)
    return labels, confidences

def get_labels(g, root_at_arr, ignore_edge_ccs):
    labels = np.full(len(root_at_arr), False)
    for v in g.nodes():
        root_at = root_at_arr[v]
        for e in g.edges(v):
            if root_at_arr[e[1]] != root_at:
                labels[v] = True
                break
    
    # Get error components to check if either side is same root_at
    subgraph_nodes = [i for i in range(len(labels)) if labels[i] == True]
    subgraph = g.subgraph(subgraph_nodes)
    ccs = list(nx.connected_components(subgraph))

    changed_ccs = []
    for cc in ccs:
        cc_neighbors = set()
        count = 0
        for v in cc:
            for n in g.adj[v]:
                if labels[n] == False:
                    cc_neighbors.add(root_at_arr[n])
                    count += 1
        if ignore_edge_ccs:
            # when there's only one neighbor (count = 1) it has to be on the edge
            # the len of neighbors set tells you if error should be ignored
            if count == 1 and (len(cc_neighbors) == 1):
                for v in cc:
                    labels[v] = False
                changed_ccs.append(cc)
        if count > 1 and (len(cc_neighbors) == 1):
            for v in cc:
                labels[v] = False
            changed_ccs.append(cc)
    return labels, changed_ccs

def get_confidences(root_at_arr, labels, changed_ccs, proofread_roots):
    
    conf = np.isin(root_at_arr, proofread_roots)
    # Confidence at errors should be True
    conf[labels == True] = True
    for cc in changed_ccs:
        for v in cc:
            conf[v] = False
    return conf