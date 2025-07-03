from auto_proof.code.pre import data_utils

import numpy as np
import time
import h5py
import networkx as nx
from typing import List, Tuple, Set

def create_labels(
    root_at_arr: np.ndarray, 
    ignore_edge_ccs: bool, 
    edges: np.ndarray, 
    proofread_roots: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates labels and confidences for a skeleton.

    This function identifies 'error' vertices and computes a confidence score for each vertex,
    indicating whether it's part of a proofread segment.

    Assumes `root_at_arr` contains bare root IDs (roots without identifiers like '_000').

    Args:
        root_at_arr: A NumPy array of shape `(N,)` where `N` is the number of
            vertices. `root_at_arr[i]` is the segmentation ID (root ID)
            of the segment that vertex `i` is located within.
        ignore_edge_ccs: A boolean flag. If True, it applies a specific rule
            to ignore certain 'error' connected components that appear to be
            at the "edge" of a segment (i.e., only have one neighbor outside the component).
        edges: A NumPy array of shape `(M, 2)` representing the edges of the
            skeleton graph. Each row `[u, v]` indicates an edge between vertex `u` and `v`.
        proofread_roots: A list of integers representing root IDs that are
            known to have been proofread.

    Returns:
        A tuple containing:
        - labels (np.ndarray): A boolean NumPy array of shape `(N,)` where `labels[i]`
            is `True` if vertex `i` is identified as an 'error' vertex, and `False` otherwise.
        - confidences (np.ndarray): A boolean NumPy array of shape `(N,)` where `confidences[i]`
            is `True` if vertex `i` is part of a proofread root or an identified error,
            and `False` otherwise (meaning it's an unproofread, non-error segment).
    """
    
    g = nx.Graph()
    g.add_nodes_from(range(len(root_at_arr)))
    g.add_edges_from(edges)
    labels, changed_ccs = get_labels(g, root_at_arr, ignore_edge_ccs)
    confidences = get_confidences(root_at_arr, labels, changed_ccs, proofread_roots)
    return labels, confidences

def get_labels(g: nx.Graph, root_at_arr: np.ndarray, ignore_edge_ccs: bool) -> Tuple[np.ndarray, List[Set[int]]]:
    """Identifies and refines 'error' labels for skeleton vertices.

    A vertex is initially labeled an 'error' if any of its adjacent vertices
    belong to a different segmentation `root_id`. These initial error labels
    are then refined by analyzing connected components of error vertices.
    Errors that are surrounded by the same component are classified as non-errors.
    Components that are at the 'edge' of a segment (only connect to one type
    of non-error neighbor) can optionally be ignored.

    Args:
        g: A NetworkX graph representing the skeleton.
        root_at_arr: A NumPy array where `root_at_arr[i]` is the segmentation ID
            (root ID) of the segment vertex `i` is located within.
        ignore_edge_ccs: A boolean flag. If True, connected components of error
            vertices that have only one type of non-error neighbor (implying
            they are "on an edge") are re-labeled as non-errors.

    Returns:
        A tuple containing:
        - labels (np.ndarray): A boolean NumPy array where `labels[i]` is `True`
            if vertex `i` is an identified error vertex, `False` otherwise.
        - changed_ccs (List[Set[int]]): A list of sets, where each set contains
            the vertex indices of connected components that were originally
            labeled as errors but were subsequently changed to non-errors
            due to the `ignore_edge_ccs` logic.
    """
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
            # When there's only one neighbor (count = 1) it has to be on the edge
            # the len of neighbors set tells you if error should be ignored
            if count == 1 and (len(cc_neighbors) == 1):
                for v in cc:
                    labels[v] = False
                changed_ccs.append(cc)
        # Doesn't count errors that are surrounded by the same component on either side.
        # The assumption being this error is very small like a spine head.
        if count > 1 and (len(cc_neighbors) == 1):
            for v in cc:
                labels[v] = False
            changed_ccs.append(cc)
    return labels, changed_ccs

def get_confidences(
    root_at_arr: np.ndarray, 
    labels: np.ndarray, 
    changed_ccs: List[Set[int]], 
    proofread_roots: List[int]
) -> np.ndarray:
    """Computes confidence scores for each vertex based on proofreading status and error labels.

    A vertex has high confidence (True) if:
    1. Its associated segment `root_id` is found in the `proofread_roots` list.
    2. It is explicitly labeled as an 'error' vertex.

    Vertices belonging to connected components whose 'error' label was
    subsequently changed (due to edge conditions in `get_labels`) are
    explicitly set to low confidence (False).

    Args:
        root_at_arr: A NumPy array where `root_at_arr[i]` is the segmentation ID
            (root ID) of the segment vertex `i` is located within.
        labels: A boolean NumPy array where `labels[i]` is `True` if vertex `i`
            is an identified 'error' vertex (after initial refinement).
        changed_ccs: A list of sets, where each set contains vertex indices of
            connected components that were re-labeled from error to non-error
            during the `get_labels` process.
        proofread_roots: A list of integers representing root IDs that are
            known to have been proofread.

    Returns:
        A boolean NumPy array of the same shape as `root_at_arr`, where `conf[i]`
        is `True` for high confidence and `False` for low confidence.
    """
    
    conf = np.isin(root_at_arr, proofread_roots)
    conf[labels == True] = True
    for cc in changed_ccs:
        for v in cc:
            conf[v] = False
    return conf