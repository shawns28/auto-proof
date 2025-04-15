from auto_proof.code.pre import data_utils

import pandas as pd
import numpy as np
from tqdm import tqdm
# import graph_tool.all as gt
import networkx as nx
import time

# class Visitor(gt.BFSVisitor):
#     """Custom class for graph tool to determine the rank using bfs"""
#     def __init__(self, vp_rank):
#         self.vp_rank = vp_rank
#         self.rank = -1

#     def examine_vertex(self, u):
#         self.rank += 1
#         self.vp_rank[u] = self.rank

def get_skel(datastack_name, skeleton_version, root, client):
    """NOTE/TODO: 
    Skeletonizes the roots and saves hdf5 files representing the cutoff number of nodes and their features
    which are pulled from cave client for the root. Additionally saves successful roots to txt.
    """
    retries = 2
    delay = 5
    root_id_without_num = int(root[:-4])
    for attempt in range(0, retries + 1):
        try: 
            skel_dict = client.skeleton.get_skeleton(root_id=root_id_without_num, skeleton_version=skeleton_version, datastack_name=datastack_name, output_format='dict')
            return True, None, skel_dict
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
                continue
            else:
                return False, e, None
    return False, None, None # Shouldn't enter

def process_skel(box_cutoff, cutoff, is_proofread, rep_coord, skel_dict):
    """NOTE/TODO: 
    Skeletonizes the roots and saves hdf5 files representing the cutoff number of nodes and their features
    which are pulled from cave client for the root. Additionally saves successful roots to txt.
    """ 

    try:
        skel_edges = np.array(skel_dict['edges'])
        skel_vertices = np.array(skel_dict['vertices'])
        skel_radii = np.array(skel_dict['radius'])
    except Exception as e:
        return False, e, None

    if not is_proofread:
        rep_index, _ = get_closest(skel_vertices, rep_coord)
    else:
        rep_index = np.random.randint(0, len(skel_vertices))

    try:
        g = nx.Graph()
        g.add_nodes_from(range(len(skel_vertices)))
        g.add_edges_from(skel_edges)
        # print("nodes order nx", g.nodes())
        # g = gt.Graph(skel_edges, directed=False)
    except Exception as e:
        return False, e, None

    skel_len = len(skel_vertices)
    rank = create_rank(g, rep_index, skel_len, box_cutoff)
    mask = rank < cutoff
    mask = np.where(mask == True)[0]

    rank = rank[mask]
    new_skel_vertices = skel_vertices[mask]
    new_skel_radii = skel_radii[mask]

    new_edges = prune_edges(mask, skel_edges)
    
    feature_dict = {}
    feature_dict['num_initial_vertices'] = len(skel_vertices)
    feature_dict['vertices'] = new_skel_vertices
    feature_dict['edges'] = new_edges
    feature_dict['radius'] = new_skel_radii
    feature_dict['rank'] = rank

    return True, None, feature_dict

def prune_edges(mask_indices, skel_edges):
    orig_to_new = {value: index for index, value in enumerate(mask_indices)}
    edge_mask = np.zeros(len(skel_edges))
    for i in range(len(skel_edges)):
        edge = skel_edges[i]
        if edge[0] in orig_to_new and edge[1] in orig_to_new:
            skel_edges[i][0] = orig_to_new[edge[0]]
            skel_edges[i][1] = orig_to_new[edge[1]]
            edge_mask[i] = 1
    return skel_edges[edge_mask.astype(bool)]

def create_rank(g, rep_index, skel_len, box_cutoff):
    rep_included = False
    while not rep_included:
        seed = np.random.randint(skel_len)
        rank_arr = bfs(g, seed, skel_len)
        # rank_arr = bfs(g, seed)
        if rank_arr[rep_index] < box_cutoff:
            rep_included = True
    return rank_arr

def bfs(g, seed_index, skel_len):
    visited = {seed_index}
    order = [seed_index]
    for _, v in nx.bfs_edges(g, seed_index):
        if v not in visited:
            visited.add(v)
            order.append(v)
    rank = {node: i for i, node in enumerate(order)}
    rank_arr = np.array([rank[i] for i in range(skel_len)])
    return rank_arr


# def bfs(g, seed_index):
#     vp_rank = g.new_vp("int")
#     gt.bfs_search(g, seed_index, Visitor(vp_rank))
#     rank_arr = vp_rank.a
#     return np.array(rank_arr)

def get_closest(arr, target):
    result = arr - target
    absit = np.abs(result)
    summed = absit.sum(axis=1)
    index = np.argmin(summed)
    value = arr[index]
    return index, value