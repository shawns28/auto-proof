from auto_proof.code.pre import data_utils
from auto_proof.code.dataset import prune_edges

import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import time
from typing import Any, Dict, Tuple, Optional # Import for type hinting

def get_skel(datastack_name: str, skeleton_version: str, root: str, client: Any) -> Tuple[bool, Optional[Exception], Optional[Dict]]:
    """Retrieves skeleton data for a given root from the CAVE client with retry logic.

    This function attempts to fetch the skeleton dictionary from the specified
    datastack and skeleton version using the CAVE client. It incorporates a
    retry mechanism to handle transient network issues or API call failures.

    Args:
        datastack_name: The name of the datastack (e.g., 'minnie65_phase3_v1').
        skeleton_version: The version of the skeletons to retrieve (e.g., '4').
        root: The root ID as a string, potentially with an identifier (e.g., '864691135335038697_000').
        client: The CAVE client object used to interact with the CAVE data.

    Returns:
        A tuple:
        - bool: True if the skeleton was successfully retrieved, False otherwise.
        - Optional[Exception]: An Exception object if an error occurred, None otherwise.
        - Optional[Dict]: The skeleton data as a dictionary if successful, None otherwise.
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

def process_skel(box_cutoff: int, cutoff: int, rep_coord: Optional[np.ndarray], skel_dict: Dict) -> Tuple[bool, Optional[Exception], Optional[Dict]]:
    """Processes raw skeleton data to extract features and prune based on connectivity.

    This function takes a raw skeleton dictionary, identifies a representative edit
    node (either by provided coordinate or randomly), performs a Breadth-First Search (BFS)
    to rank nodes by distance from the representative, and then prunes the skeleton
    to include only nodes within a specified connectivity cutoff.

    Args:
        box_cutoff: The maximum BFS rank (distance from representative edit) for a randomly chosen seed
                    to be considered valid. This helps ensure the representative edit is somewhat central.
        cutoff: The maximum BFS rank from the final chosen representative to include nodes in the
                processed skeleton. Nodes beyond this rank are pruned.
        rep_coord: A NumPy array representing the 3D coordinate of the representative point
                   for the neuron, or None if it's a proofread root (in which case a random
                   representative will be chosen from within the skeleton).
        skel_dict: A dictionary containing raw skeleton data, expected to have 'edges',
                   'vertices', and 'radius' keys.

    Returns:
        A tuple:
        - bool: True if the skeleton was successfully processed, False otherwise.
        - Optional[Exception]: An Exception object if an error occurred, None otherwise.
        - Optional[Dict]: A dictionary containing processed features ('num_initial_vertices',
                          'vertices', 'edges', 'radius', 'rank') if successful, None otherwise.
    """ 

    try:
        skel_edges = np.array(skel_dict['edges'])
        skel_vertices = np.array(skel_dict['vertices'])
        skel_radii = np.array(skel_dict['radius'])

    except Exception as e:
        return False, e, None

    if rep_coord is not None:
        rep_index, _ = get_closest(skel_vertices, rep_coord)
    else:
        rep_index = np.random.randint(0, len(skel_vertices))

    try:
        g = nx.Graph()
        g.add_nodes_from(range(len(skel_vertices)))
        g.add_edges_from(skel_edges)
        skel_len = len(skel_vertices)
        rank = create_rank(g, rep_index, skel_len, box_cutoff)
        mask = np.where(rank < cutoff)[0]
        rank = rank[mask]
        new_skel_vertices = skel_vertices[mask]
        new_skel_radii = skel_radii[mask]
            
    except Exception as e:
        return False, e, None

    new_edges = prune_edges(skel_edges, mask)
    
    feature_dict = {}
    feature_dict['num_initial_vertices'] = len(skel_vertices)
    feature_dict['vertices'] = new_skel_vertices
    feature_dict['edges'] = new_edges
    feature_dict['radius'] = new_skel_radii
    feature_dict['rank'] = rank

    return True, None, feature_dict

def create_rank(g: nx.Graph, rep_index: int, skel_len: int, box_cutoff: int) -> np.ndarray:
    """Ensures a representative index is within a 'well-connected' region and generates BFS ranks.

    This function repeatedly performs a Breadth-First Search (BFS) starting from a random
    seed node until a seed is found such that the `rep_index` (the actual representative
    node for the skeleton) is within `box_cutoff` BFS steps from that random seed.

    Args:
        g: A NetworkX graph representing the skeleton.
        rep_index: The index of the chosen representative node within the skeleton.
        skel_len: The total number of vertices in the skeleton.
        box_cutoff: The maximum allowed BFS rank for `rep_index` from a randomly
                    chosen seed for the seed to be considered valid.

    Returns:
        np.ndarray: A NumPy array where `rank_arr[i]` is the BFS distance (rank) of node `i`
                    from the selected random seed.
    """
    rep_included = False
    while not rep_included:
        seed = np.random.randint(skel_len)
        rank_arr = bfs(g, seed, skel_len)
        if rank_arr[rep_index] < box_cutoff:
            rep_included = True
    return rank_arr

def bfs(g: nx.Graph, seed_index: int, skel_len: int) -> np.ndarray:
    """Performs a Breadth-First Search (BFS) starting from a seed node and returns node ranks.

    The rank of a node is its shortest path distance (number of edges) from the seed node.
    
    Args:
        g: A NetworkX graph.
        seed_index: The starting node for the BFS.
        skel_len: The total number of nodes in the graph.

    Returns:
        np.ndarray: A NumPy array where `rank_arr[i]` is the BFS rank of node `i` from
                    `seed_index`.
    """
    visited = {seed_index}
    order = [seed_index]
    for _, v in nx.bfs_edges(g, seed_index):
        if v not in visited:
            visited.add(v)
            order.append(v)
    rank = {node: i for i, node in enumerate(order)}
    rank_arr = np.array([rank[i] for i in range(skel_len)])
    return rank_arr

def get_closest(arr: np.ndarray, target: np.ndarray) -> Tuple[int, np.ndarray]:
    """Finds the index and value of the row in a 2D array that is closest to a target coordinate.

    Closeness is determined by Manhattan distance (sum of absolute differences across dimensions).

    Args:
        arr: A NumPy array of shape (N, D), where N is the number of points and D is the dimension.
        target: A NumPy array of shape (D,), representing the target coordinate.

    Returns:
        A tuple:
        - int: The index of the row in `arr` that is closest to `target`.
        - np.ndarray: The actual coordinate (row) from `arr` that is closest to `target`.
    """
    result = arr - target
    absit = np.abs(result)
    summed = absit.sum(axis=1)
    index = np.argmin(summed)
    value = arr[index]
    return index, value