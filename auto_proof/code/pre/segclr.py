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
from typing import Dict, Any, List, Tuple, Optional, Union # Import for type hinting

def get_segclr_emb(
    root: str, 
    vertices: np.ndarray, 
    edges: np.ndarray, 
    root_at_arr: np.ndarray, 
    embedding_reader: EmbeddingReader, 
    emb_dim: int, 
    visualize_radius: bool, 
    small_radius: float, 
    large_radius: float
) -> Tuple[bool, Optional[Exception], Optional[np.ndarray], Optional[np.ndarray]]:
    """Retrieves and processes SegCLR embeddings for skeleton vertices.

    This function queries a pre-loaded `EmbeddingReader` for embeddings corresponding
    to the `root_at_arr` (root at appropriate materialization version) of each skeleton vertex.
    It then assigns these embeddings to the skeleton vertices, handling cases where embeddings are
    not found or where vertices are too far from available embedding coordinates
    within their respective segments. It incorporates a dual-radius search strategy
    and can optionally generate a visualization of the radius search outcome.

    Args:
        root: The root ID of the current skeleton (e.g., '864691135335038697_000'). 
        Used for logging and visualization filename.
        vertices: A NumPy array of shape `(N, 3)` representing the 3D coordinates
            of N skeleton vertices.
        edges: A NumPy array of shape `(M, 2)` representing the edges of the skeleton graph.
        root_at_arr: A NumPy array of shape `(N,)` where `root_at_arr[i]` is the
            root at mat version that vertex `i` belongs to.
        embedding_reader: An initialized `EmbeddingReader` object, capable of fetching
            embeddings for given segment IDs.
        emb_dim: An integer specifying the expected dimension of the embeddings.
        visualize_radius: A boolean flag. If True, a visualization HTML file is
            generated showing which vertices fell within which search radius.
        small_radius: A float, the smaller radius for querying embeddings. Vertices
            within this radius of an embedding coordinate will use the average of
            embeddings within this radius.
        large_radius: A float, the larger radius for querying embeddings. If no
            embeddings are found within `small_radius`, and the vertex's degree
            is high (or it's a leaf node), a search up to `large_radius` is performed.

    Returns:
        A tuple:
        - bool: True if embeddings were processed successfully, False otherwise.
        - Optional[Exception]: An Exception object if an error occurred during processing,
            None otherwise.
        - Optional[np.ndarray]: A NumPy array of shape `(N, emb_dim)` containing the
            assigned SEGCLR embeddings for each vertex, or None if failed.
        - Optional[np.ndarray]: A NumPy array of shape `(N,)` where `has_emb[i]` is
            1 if vertex `i` was successfully assigned an embedding, and 0 otherwise.
            Returns None if failed.
    """
    try: 
        g = nx.Graph()
        g.add_nodes_from(range(len(vertices)))
        g.add_edges_from(edges)

        roots_at_to_indices = create_root_at_dict(root_at_arr)

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
            emb_vals = []
            coords = []
            for coord_key, emb_val in embs.items():
                emb_vals.append(emb_val)
                coords.append(coord_key)
            emb_vals = np.array(emb_vals)
            coords = np.array(coords)


            tree = cKDTree(coords)

            for index in roots_at_to_indices[root]:

                vertice = vertices[index]
                
                q_indices = tree.query_ball_point(vertice, small_radius)
                if len(q_indices) == 0:
                    too_far_for_large = True
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
                else:
                    result[index] = np.mean([emb_vals[q_ind][:emb_dim] for q_ind in q_indices], axis=0)            
        all_zeros = np.all(result == 0, axis=1)
        zero_indices = np.where(all_zeros)[0]
        if len(zero_indices) > 0:
            has_emb[zero_indices] = 0
        if visualize_radius:
            path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/segclr_test/{original_root}_{small_radius}_{large_radius}radius_segclr.html'
            visualize_segclr(vertices, edges, coords, too_far_small_radius_indices, too_far_large_radius_indices, path)
        return True, None, result, has_emb
    except Exception as e:
        return False, e, None, None

def create_root_at_dict(roots_at: np.ndarray) -> Dict[int, List[int]]:
    """Creates a dictionary mapping segmentation root_at root IDs to lists of vertex indices.

    It returns a dictionary where keys are unique root at IDs and values are lists
    of integer indices of skeleton vertices that fall within that specific root at ID.

    Args:
        roots_at: A NumPy array of integers, where `roots_at[i]` is the segmentation
            ID for the i-th skeleton vertex.

    Returns:
        A dictionary where keys are root at IDs and values are
        lists of integer indices (from the original `roots_at` array) 
        that belong to that root ID.
    """
    roots_at_to_indices = {}
    for i in range(len(roots_at)):
        root = int(roots_at[i])
        if root not in roots_at_to_indices:
            roots_at_to_indices[root] = []
        roots_at_to_indices[root].append(i)
    return roots_at_to_indices

def sharder(segment_id: int, num_shards: int, bytewidth: int) -> int:
    """Computes a shard ID for a given segment ID using an MD5 hash.

    This function utilizes the `md5_shard` utility to determine which shard
    a `segment_id` belongs to, based on the total number of shards and a
    specified bytewidth.

    Args:
        segment_id: The integer ID of the segment.
        num_shards: The total number of available shards.
        bytewidth: The byte width to use for the MD5 hash calculation.

    Returns:
        An integer representing the shard ID for the given segment ID.
    """
    return md5_shard(segment_id, num_shards=num_shards, bytewidth=bytewidth)

def get_roots_at_seglcr_version(root: str, data_dir: str, mat_versions: List[str], is_whole_cell: bool) -> str:
    """Determines the appropriate materialization version for SEGCLR data based on root ID history.

    This function attempts to identify which materialization version a given `root` 
    (without its identifier suffix, e.g., '864691135335038697' from '864691135335038697_000')
    should be associated with. If the root is a fully proofread root then we use that as
    the SegCLR version. Otherwise, we use the closest previous SegCLR version. 
    This is crucial for retrieving the correct SegCLR embeddings, as embeddings
    are version-dependent. The `mat_versions` list is expected to contain at least
    three materialization version strings.

    For example if a new root was created inbetween 943 and 1300 materialization versions
    then it would use the 943 SegCLR version. If the root was a fully proofread root at 943,
    we would use the 943 SegCLR version.

    Args:
        root: The full root ID string (e.g., '12345_000').
        data_dir: The base directory where root lists and other data are stored.
        mat_versions: A list of at least three strings, representing different
            materialization versions. For example, `['v1', 'v2', 'v3']`.
            The function specifically expects `mat_versions[0]`, `mat_versions[1]`,
            and `mat_versions[2]`.
        is_whole_cell: A boolean flag indicating whether the root is part of a
            "whole cell" prediction pipeline that has never gotten an edit or
            the edit is outside of the specified mat versions. 
            If True and the root is not found in any lists, it defaults to `mat_version1`.
            If False, an Exception is raised.

    Returns:
        A string representing the determined materialization version for the given root.

    Raises:
        Exception: If `is_whole_cell` is False and the `root` (without identifier)
            is not found in any of the provided root lists.
    """
    mat_version1 = mat_versions[0]
    mat_version2 = mat_versions[1]
    mat_version3 = mat_versions[2]
    roots1 = data_utils.load_txt(f'{data_dir}roots_{mat_version1}_{mat_version3}/roots_{mat_version1}_{mat_version2}.txt')
    roots2 = data_utils.load_txt(f'{data_dir}proofread/{mat_version2}_unique.txt')
    roots3 = data_utils.load_txt(f'{data_dir}roots_{mat_version1}_{mat_version3}/post_edit_roots.txt')
    roots4 = data_utils.load_txt(f'{data_dir}proofread/{mat_version3}_unique.txt')

    root_without_ident = root[:-4]
    if root_without_ident in roots1:
        segmentation_version = mat_version1
    elif root_without_ident in roots2 or root_without_ident in roots3:
        segmentation_version = mat_version2
    elif root_without_ident in roots4:
        segmentation_version = mat_version3
    else:
        if is_whole_cell:
            print("Root not in any of the root lists")
            segmentation_version = mat_version1
        else:
            raise Exception("Root not in any of the root lists")
    return segmentation_version