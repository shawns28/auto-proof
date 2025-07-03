from auto_proof.code.pre import data_utils

import numpy as np
from tqdm import tqdm
import h5py
import networkx as nx
import multiprocessing
from collections import deque
import argparse
import os
import time
import re

def main():
    """Executes the complete workflow for creating and splitting root groups.

    This function orchestrates the entire process, including:
    1. Loading configuration parameters.
    2. Identifying and filtering a set of 'roots' based on various criteria.
    3. Creating a mapping from each root to its 'latest roots' (roots at the latest
       materialization version).
    4. Constructing a graph where 'latest roots' are connected if they share a common
       original root.
    5. Identifying connected components within this graph.
    6. Mapping original roots to their corresponding connected component identifiers.
    7. Grouping roots by their connected component.
    8. (Optional) Identifying 'no error in box' roots based on feature and label data.
    9. Splitting the root groups into training, validation, and testing sets
       while maintaining distribution of proofread roots.
    """
    data_config = data_utils.get_config('data')
    client_config = data_utils.get_config('client')
    parser = argparse.ArgumentParser(description="Process root data for splitting.")
    parser.add_argument("-n", "--num_processes", type=int,
                        help="Number of processes to use for multiprocessing.")
    args = parser.parse_args()

    if args.num_processes:
        data_config['multiprocessing']['num_processes'] = args.num_processes

    data_dir = data_config['data_dir']
    # The `create_client` function from `data_utils` is expected to return
    # client-related objects and version information.
    _, _, mat_version_start, mat_version_end = data_utils.create_client(client_config)

    roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'
    features_dir = f'{data_dir}{data_config['features']['features_dir']}'
    labels_dir = f'{data_dir}{data_config['labels']['labels_at_latest_dir']}' \
                 f'{data_config['labels']['latest_mat_version']}_' \
                 f'{data_config['labels']['labels_type']}/'

    post_label_roots = data_utils.load_txt(
        f'{roots_dir}{data_config['labels']['post_label_roots']}')
    print("post_label_roots len", len(post_label_roots))
    
    post_segclr_roots = data_utils.load_txt(
        f'{roots_dir}{data_config['segclr']['post_segclr_roots']}')
    print("post_segclr_roots len", len(post_segclr_roots))
    roots = np.intersect1d(post_label_roots, post_segclr_roots)
    print("roots combined len", len(roots))

    # NOTE: This is due to SegCLR not existing for 1300
    roots_1300_unique_copied = f'{data_dir}{data_config['proofread']['proofread_dir']}1300_unique_copied.txt'
    roots = np.setdiff1d(roots, roots_1300_unique_copied)
    print("final roots len", len(roots))

    # TODO: Remove after testing
    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/all_roots_og.txt")

    split = data_config['split']['split']
    split_dir = f'{roots_dir}{data_config['split']['split_dir']}{len(roots)}/'
    split_dicts_dir = f'{split_dir}dicts/'

    # Ensure necessary directories exist.
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(split_dicts_dir, exist_ok=True)

    num_processes = data_config['multiprocessing']['num_processes']

    # Step 1: Create or load mapping from original root to its 'latest roots'.
    root_to_latest_set_path = f'{split_dicts_dir}root_to_latest_set.pkl'
    if not os.path.exists(root_to_latest_set_path):
        print("creating root to latest set...")
        roots_at_latest_dir = f'{data_dir}{data_config['labels']['roots_at_latest_dir']}' \
                               f'{data_config['labels']['latest_mat_version']}/'
        root_to_latest_set = create_latest_set_dict(
            roots, roots_at_latest_dir, num_processes, root_to_latest_set_path)
        print("root to latest set key len", len(root_to_latest_set.keys()))
    else:
        print("loading root to latest set...")
        root_to_latest_set = data_utils.load_pickle_dict(root_to_latest_set_path)

    # Step 2: Create a graph where shared "roots at latest" get edges connecting them.
    print("Creating graph...")
    graph = create_graph(root_to_latest_set)

    # Step 3 & 4: Get connected components and map 'latest roots' to CC identifiers.
    # The `nx.connected_components` function returns an iterator of sets, where each set
    # contains the nodes of a connected component.
    root_latest_to_cc = {}
    for i, component in enumerate(nx.connected_components(graph)):
        for node in component:
            root_latest_to_cc[int(node)] = i

    root_latest_to_cc_path = f'{split_dicts_dir}root_latest_to_cc.pkl'
    if not os.path.exists(root_latest_to_cc_path):
        print("Creating root_latest_to_cc dict...")
        data_utils.save_pickle_dict(root_latest_to_cc_path, root_latest_to_cc)
    else:
        print("Loading root_latest_to_cc dict...")
        root_latest_to_cc = data_utils.load_pickle_dict(root_latest_to_cc_path)

    # Step 5: Map original roots to their connected component identifiers.
    root_to_cc_path = f'{split_dicts_dir}root_to_cc.pkl'
    if not os.path.exists(root_to_cc_path):
        print("Creating root_to_cc dict...")
        root_to_cc = create_root_to_cc(root_latest_to_cc, root_to_latest_set, root_to_cc_path)
        print(f"Number of roots in root_to_cc: {len(root_to_cc)}")
    else:
        print("Loading root_to_cc dict...")
        root_to_cc = data_utils.load_pickle_dict(root_to_cc_path)

     # Step 6: Map connected component identifiers to groups of original roots.
    cc_to_root_group_path = f'{split_dicts_dir}cc_to_root_group.pkl'
    if not os.path.exists(cc_to_root_group_path):
        print("Creating cc_to_root_group dict...")
        cc_to_root_group = create_cc_to_root_group(root_to_cc, cc_to_root_group_path)
    else:
        print("Loading cc_to_root_group dict...")
        cc_to_root_group = data_utils.load_pickle_dict(cc_to_root_group_path)
    
    # Step 7: Create a list of root groups, where each group corresponds to a CC.
    cc_root_group_list_path = f'{split_dicts_dir}cc_root_group_list.pkl'
    if not os.path.exists(cc_root_group_list_path):
        print("Creating cc_root_group_list...")
        cc_root_group_list = create_cc_root_group_list(
            cc_to_root_group, cc_root_group_list_path)
    else:
        print("Loading cc_root_group_list...")
        cc_root_group_list = data_utils.load_pickle_dict(cc_root_group_list_path)

    # Step 8 (Optional): Determine 'no error in box' roots.
    no_error_in_box_dict_path = f'{split_dicts_dir}no_error_in_box_dict.pkl'
    if not os.path.exists(no_error_in_box_dict_path):
        print("Creating no error in box dict...")
        no_error_in_box_dict = create_conf_no_error_in_box_roots(
            roots, features_dir, labels_dir, data_config['features']['box_cutoff'],
            num_processes, no_error_in_box_dict_path)
    else:
        print("Loading no error in box dict...")
        no_error_in_box_dict = data_utils.load_pickle_dict(no_error_in_box_dict_path)

    # Step 9: Create split
    # Load proofread roots.
    proofread_config = data_config['proofread']
    proofread_mat_version1 = proofread_config['mat_versions'][0]
    proofread_mat_version2 = proofread_config['mat_versions'][1]
    proofread_roots = data_utils.load_txt(
        f'{data_dir}{proofread_config['proofread_dir']}'
        f'{proofread_mat_version1}_{proofread_mat_version2}.txt')
    # Append '_000' suffix to proofread roots for consistency with other root IDs.
    proofread_roots = [str(root) + '_000' for root in proofread_roots]

    # TODO: Remove the hi
    # Define paths for split root lists.
    val_roots_path = f'{split_dir}val_roots_hi.txt'
    train_roots_path = f'{split_dir}train_roots_hi.txt'
    test_roots_path = f'{split_dir}test_roots_hi.txt'

    # Create the train/validation/test split if files don't exist.
    if not (os.path.exists(val_roots_path) and os.path.exists(train_roots_path)):
        print("Creating split...")
        data_utils.save_txt(f'{split_dir}all_roots_hi.txt', roots)
        start_time = time.time()
        # 'default' flag controls sorting logic for connected components.
        default_split_logic = True
        train_roots, val_roots, test_roots = create_split(
            no_error_in_box_dict, cc_root_group_list, proofread_roots,
            split, split_dir, default_split_logic)
        end_time = time.time()
        print(f"Creating split took {end_time - start_time:.2f} seconds.")

        print("Checking split distribution of proofread roots...")
        check_proofread_dist(proofread_roots, train_roots, val_roots, test_roots)
    else:
        print("Loading validation, test and training roots from existing files...")
        val_roots = data_utils.load_txt(val_roots_path)
        train_roots = data_utils.load_txt(train_roots_path)
        test_roots = data_utils.load_txt(test_roots_path)
    
    # Create and save lists of 'no error in box' roots for each split.
    val_conf_no_error_in_box_roots_list_path = \
        f'{split_dir}val_conf_no_error_in_box_roots_hi.txt'
    train_conf_no_error_in_box_roots_list_path = \
        f'{split_dir}train_conf_no_error_in_box_roots_hi.txt'
    test_conf_no_error_in_box_roots_list_path = \
        f'{split_dir}test_conf_no_error_in_box_roots_hi.txt'

    if not (os.path.exists(val_conf_no_error_in_box_roots_list_path) and
            os.path.exists(train_conf_no_error_in_box_roots_list_path)):
        print("Creating validation 'no error in box' roots list...")
        create_conf_no_error_in_box_roots_list(
            no_error_in_box_dict, val_roots, val_conf_no_error_in_box_roots_list_path)
        print("Creating training 'no error in box' roots list...")
        create_conf_no_error_in_box_roots_list(
            no_error_in_box_dict, train_roots, train_conf_no_error_in_box_roots_list_path)
        print("Creating test 'no error in box' roots list...")
        create_conf_no_error_in_box_roots_list(
            no_error_in_box_dict, test_roots, test_conf_no_error_in_box_roots_list_path)

def create_latest_set_dict(roots, roots_at_latest_dir, num_processes, save_path):
    """Creates a dictionary mapping each original root to a set of its 'latest roots'.

    Args:
        roots: A list of original root identifiers.
        roots_at_latest_dir: Directory containing HDF5 files for 'roots at latest'.
        num_processes: Number of parallel processes to use for loading data.
        save_path: File path to save the generated dictionary as a pickle file.

    Returns:
        A dictionary where keys are original root IDs and values are sets of
        their corresponding 'latest roots'.
    """
    root_to_latest_set = {}
    args_list = [(root, roots_at_latest_dir) for root in roots]

    # Use multiprocessing to efficiently load 'latest roots' for each original root.
    with multiprocessing.Pool(processes=num_processes) as pool, \
         tqdm(total=len(roots), desc="Loading latest roots") as pbar:
        for latest_roots, root in pool.imap_unordered(__get_latest_roots__, args_list):
            root_to_latest_set[root] = set(latest_roots)
            pbar.update()

    data_utils.save_pickle_dict(save_path, root_to_latest_set)
    return root_to_latest_set

def __get_latest_roots__(data):
    """Helper function to load 'latest roots' for a given root from an HDF5 file.

    This function is designed to be used with multiprocessing.

    Args:
        data: A tuple containing (root_id, roots_at_latest_dir).

    Returns:
        A tuple containing (latest_roots_array, root_id).
    """
    root, roots_at_latest_dir = data
    roots_at_path = f'{roots_at_latest_dir}{root}.hdf5'
    with h5py.File(roots_at_path, 'r') as roots_at_f:
        latest_roots = roots_at_f['roots_at'][:]
    return latest_roots, root

def create_graph(root_to_latest_set):
    """Creates a NetworkX graph where 'latest roots' are nodes and edges connect
    'latest roots' that originated from the same original root.

    Args:
        root_to_latest_set: A dictionary mapping original root IDs to sets of
                            their 'latest roots'.

    Returns:
        A NetworkX Graph object.
    """
    graph = nx.Graph()
    edges_to_add = []

    for _, latest_roots_set in root_to_latest_set.items():
        # Convert set to list for ordered iteration to create edges.
        curr_nodes = [str(node) for node in latest_roots_set]
        if len(curr_nodes) == 1:
            # If an original root only maps to one 'latest root', add it as a single node.
            graph.add_node(curr_nodes[0])
        else:
            # Create edges between consecutive 'latest roots' from the same original root.
            pairs = [(curr_nodes[i], curr_nodes[i + 1]) for i in range(len(curr_nodes) - 1)]
            edges_to_add.extend(pairs)

    graph.add_edges_from(edges_to_add)
    return graph

def create_root_to_cc(root_latest_to_cc, root_to_latest_set, root_to_cc_path):
    """Creates a dictionary mapping each original root to its connected component identifier.

    This mapping is derived by taking the first 'latest root' associated with an
    original root and finding its component ID. This assumes that all 'latest roots'
    for a given original root belong to the same connected component, which should
    be true by design of `create_graph`.

    Args:
        root_latest_to_cc: Dictionary mapping 'latest root' IDs to CC IDs.
        root_to_latest_set: Dictionary mapping original root IDs to sets of
                            their 'latest roots'.
        root_to_cc_path: File path to save the generated dictionary.

    Returns:
        A dictionary where keys are original root IDs and values are their
        corresponding connected component IDs.
    """
    root_to_cc = {}
    for root, latest_roots_set in root_to_latest_set.items():
        # Get an arbitrary element from the set of latest roots.
        # It is assumed that all latest roots for a given root belong to the same CC.
        if latest_roots_set: # Ensure the set is not empty
            root_latest = next(iter(latest_roots_set))
            root_to_cc[root] = root_latest_to_cc[root_latest]
        else:
            # Handle cases where an original root has no 'latest roots' if possible.
            # This might indicate an issue in the upstream `root_to_latest_set` creation.
            print(f"Warning: Root {root} has no associated 'latest roots'. Skipping.")
    data_utils.save_pickle_dict(root_to_cc_path, root_to_cc)
    return root_to_cc

def create_cc_to_root_group(root_to_cc, cc_to_root_group_path):
    """Creates a dictionary mapping each connected component identifier to a list
    of original roots that belong to that component.

    Args:
        root_to_cc: Dictionary mapping original root IDs to their CC IDs.
        cc_to_root_group_path: File path to save the generated dictionary.

    Returns:
        A dictionary where keys are CC IDs and values are lists of original
        root IDs belonging to that component.
    """
    cc_to_root_group = {}
    for root, cc in root_to_cc.items():
        if cc not in cc_to_root_group:
            cc_to_root_group[cc] = []
        cc_to_root_group[cc].append(root)
    data_utils.save_pickle_dict(cc_to_root_group_path, cc_to_root_group)
    return cc_to_root_group

def create_cc_root_group_list(cc_to_root_group, cc_root_group_list_path):
    """Converts the dictionary of CCs to root groups into a list of root groups.

    Args:
        cc_to_root_group: Dictionary mapping CC IDs to lists of roots.
        cc_root_group_list_path: File path to save the generated list.

    Returns:
        A list of lists, where each inner list represents a group of original
        roots belonging to the same connected component.
    """
    cc_root_group_list = []
    # Iterating over .values() directly gives the lists of roots.
    for root_group in cc_to_root_group.values():
        cc_root_group_list.append(root_group)
    data_utils.save_pickle_dict(cc_root_group_list_path, cc_root_group_list)
    return cc_root_group_list

def create_split(no_error_in_box_dict, cc_root_group_list, proofread_roots, split_ratios,
                 split_dir, default_split_logic):
    """Splits the connected component root groups into training, validation,
    and testing sets based on specified ratios and a sorting strategy.

    The primary goal is to distribute 'proofread' roots and total roots
    proportionally across the splits.

    Args:
        no_error_in_box_dict: Dictionary indicating if a root has 'no error in box'.
        cc_root_group_list: A list of lists, where each inner list is a group of
                            roots belonging to a connected component.
        proofread_roots: A list of roots identified as proofread.
        split_ratios: A list or tuple of three floats representing the desired
                      proportions for train, validation, and test sets (e.g., [0.8, 0.1, 0.1]).
        split_dir: Directory where the split root lists will be saved.
        default_split_logic: If True, sort groups by the number of proofread roots
                             (descending). If False, sort by the number of
                             'no error in box' roots (ascending - this is
                             marked as "non-default logic" in the original code).

    Returns:
        A tuple containing three numpy arrays: (train_roots, val_roots, test_roots).
    """
    # Sort the connected component root groups based on the specified logic.
    if default_split_logic:
        # Sort by the count of proofread roots in descending order.
        root_list_sorted = sorted(
            cc_root_group_list,
            key=lambda sublist: np.intersect1d(proofread_roots, np.array(sublist)).size,
            reverse=True
        )
    else:
        # Sort by the count of 'no error in box' roots in ascending order (non-default).
        root_list_sorted = sorted(
            cc_root_group_list,
            key=lambda sublist: sum(1 for root in sublist if no_error_in_box_dict.get(root, False)),
        )

    size_list = [len(arr) for arr in root_list_sorted]
    proofread_count_list = [np.intersect1d(proofread_roots, np.array(cc_root_group)).size
                            for cc_root_group in root_list_sorted]

    print(f"First 15 proofread counts of sorted groups: {proofread_count_list[:15]}")
    print(f"First 15 sizes of sorted groups: {size_list[:15]}")
    print(f"Sample group lengths: {len(root_list_sorted[0])}, {len(root_list_sorted[1])}, "
          f"{len(root_list_sorted[2])}, {len(root_list_sorted[10])}, {len(root_list_sorted[-1])}")

    # Use deque for efficient pop operations from the beginning of the list.
    root_list_deque = deque(root_list_sorted)
    size_list_deque = deque(size_list)
    proofread_count_list_deque = deque(proofread_count_list)

    # Initialize lists to hold roots for each split and their cumulative counts.
    sum_size_proportions = [0 for _ in split_ratios]
    sum_proofread_proportions = [0 for _ in split_ratios]
    split_roots = [[] for _ in split_ratios]

    # Distribute groups with proofread roots first, prioritizing the train split.
    # The loop continues as long as there are groups and the first group has proofread roots.
    while root_list_deque and proofread_count_list_deque[0] > 0:
        # Add the largest (or most proofread, depending on sort) group to the first split (train).
        split_roots[0].extend(root_list_deque.popleft())
        sum_size_proportions[0] += size_list_deque.popleft()
        sum_proofread_proportions[0] += proofread_count_list_deque.popleft()

        # Balance the distribution among splits based on proofread root proportions.
        # This nested loop attempts to balance the current proportions by adding groups
        # to other splits if they are falling behind their target ratio relative to the train split.
        while (root_list_deque and proofread_count_list_deque[0] > 0 and
               (sum_proofread_proportions[1] / (sum_proofread_proportions[0] or 1)) *
               (split_ratios[0] / split_ratios[1]) < 1 and
               (sum_proofread_proportions[2] / (sum_proofread_proportions[0] or 1)) *
               (split_ratios[0] / split_ratios[2]) < 1):
            if root_list_deque and proofread_count_list_deque[0] > 0:
                for i in range(1, len(split_ratios)): # Iterate over val and test splits
                    if root_list_deque and proofread_count_list_deque[0] > 0:
                        split_roots[i].extend(root_list_deque.popleft())
                        sum_size_proportions[i] += size_list_deque.popleft()
                        sum_proofread_proportions[i] += proofread_count_list_deque.popleft()
                    else:
                        break # No more roots to distribute
            else:
                break # No more roots with proofread counts to distribute

    print(f"Split roots length after proofread distribution: "
          f"Train: {len(split_roots[0])}, Val: {len(split_roots[1])}, Test: {len(split_roots[2])}")

    # Distribute remaining groups (those without proofread roots, or after initial balancing)
    # based on total root count proportions.
    while root_list_deque:
        split_roots[0].extend(root_list_deque.popleft())
        sum_size_proportions[0] += size_list_deque.popleft()
        sum_proofread_proportions[0] += proofread_count_list_deque.popleft() # Still update, but will be 0 for these

        while (root_list_deque and
               (sum_size_proportions[1] / (sum_size_proportions[0] or 1)) *
               (split_ratios[0] / split_ratios[1]) < 1 and
               (sum_size_proportions[2] / (sum_size_proportions[0] or 1)) *
               (split_ratios[0] / split_ratios[2]) < 1):
            if root_list_deque:
                for i in range(1, len(split_ratios)):
                    if root_list_deque:
                        split_roots[i].extend(root_list_deque.popleft())
                        sum_size_proportions[i] += size_list_deque.popleft()
                        sum_proofread_proportions[i] += proofread_count_list_deque.popleft()
                    else:
                        break
            else:
                break

    print(f"Final proportions for proofread roots: {sum_proofread_proportions}")
    print(f"Final proportions for total roots (size): {sum_size_proportions}")
    print(f"Split roots final lengths: "
          f"Train: {len(split_roots[0])}, Val: {len(split_roots[1])}, Test: {len(split_roots[2])}")

    for i, sub_split_roots in enumerate(split_roots):
        conf_count = sum(1 for root in sub_split_roots if no_error_in_box_dict.get(root, False))
        print(f"Split {i} 'no error in box' root count: {conf_count}")

    # Sanity check: count proofread roots in the first split (train).
    print(f"Proofread roots in train split: "
          f"{np.intersect1d(np.array(proofread_roots), np.array(split_roots[0])).size}")

    # Save the split root lists to text files.
    data_utils.save_txt(f'{split_dir}train_roots_hi.txt', split_roots[0])
    data_utils.save_txt(f'{split_dir}val_roots_hi.txt', split_roots[1])
    data_utils.save_txt(f'{split_dir}test_roots_hi.txt', split_roots[2])

    # Save split metadata.
    with open(f'{split_dir}split_metadata.txt', 'w') as meta_f:
        meta_f.write(f'train_{sum_size_proportions[0]}_{sum_proofread_proportions[0]}\n')
        meta_f.write(f'val_{sum_size_proportions[1]}_{sum_proofread_proportions[1]}\n')
        meta_f.write(f'test_{sum_size_proportions[2]}_{sum_proofread_proportions[2]}\n')

    return np.array(split_roots[0]), np.array(split_roots[1]), np.array(split_roots[2])

def check_proofread_dist(proofread_roots, train_roots, val_roots, test_roots):
    """Checks and prints the distribution of proofread roots across the splits.

    Args:
        proofread_roots: A list or array of all proofread root IDs.
        train_roots: A list or array of root IDs in the training set.
        val_roots: A list or array of root IDs in the validation set.
        test_roots: A list or array of root IDs in the testing set.
    """
    train_count = np.intersect1d(proofread_roots, train_roots).size
    val_count = np.intersect1d(proofread_roots, val_roots).size
    test_count = np.intersect1d(proofread_roots, test_roots).size

    total_proofread = len(proofread_roots)
    print(f"Proofread count in Train/Val/Test: {train_count}, {val_count}, {test_count}")
    print(f"Proofread ratio in Train/Val/Test: "
          f"{train_count / total_proofread:.4f}, "
          f"{val_count / total_proofread:.4f}, "
          f"{test_count / total_proofread:.4f}")

def create_conf_no_error_in_box_roots(roots, features_dir, labels_dir, box_cutoff,
                                      num_processes, conf_no_error_in_box_roots_path):
    """Determines which roots have 'no error in box' based on confidence and labels.

    Args:
        roots: A list of root identifiers to check.
        features_dir: Directory containing HDF5 feature files.
        labels_dir: Directory containing HDF5 label files.
        box_cutoff: A threshold used to filter data based on 'rank'.
        num_processes: Number of parallel processes to use for checking roots.
        conf_no_error_in_box_roots_path: File path to save the dictionary
                                         of 'no error in box' roots.

    Returns:
        A dictionary where keys are root IDs and values are booleans (True if
        'no error in box', False otherwise).
    """
    args_list = [(root, features_dir, labels_dir, box_cutoff) for root in roots]
    conf_no_error_in_box = {}

    with multiprocessing.Pool(processes=num_processes) as pool, \
         tqdm(total=len(roots), desc="Checking 'no error in box' roots") as pbar:
        for root, root_bool in pool.imap_unordered(__check_root__, args_list):
            conf_no_error_in_box[root] = root_bool
            pbar.update()

    data_utils.save_pickle_dict(conf_no_error_in_box_roots_path, conf_no_error_in_box)
    return conf_no_error_in_box


def create_conf_no_error_in_box_roots_list(conf_no_error_in_box, roots,
                                            conf_no_error_in_box_roots_list_path):
    """Filters a list of roots to include only those identified as 'no error in box'
    and saves the filtered list.

    Additionally, it filters for roots ending with '_000', to filter out root copies.

    Args:
        conf_no_error_in_box: Dictionary indicating if a root has 'no error in box'.
        roots: The list of roots to filter.
        conf_no_error_in_box_roots_list_path: File path to save the filtered list.
    """
    conf_list = [root for root in roots if conf_no_error_in_box.get(root, False)]

    print(f"Confidence 'no error in box' list length before filtering: {len(conf_list)}")
    # Filter roots that end with '_000', typically indicating original roots.
    filtered_roots = [root for root in conf_list if re.search(r'_000$', root)]
    print(f"Confidence 'no error in box' list length after filtering ('_000' suffix): "
          f"{len(filtered_roots)}")

    data_utils.save_txt(conf_no_error_in_box_roots_list_path, filtered_roots)

def __check_root__(data):
    """Helper function to check if a single root has 'no error in box'.

    'No error in box' is determined by examining 'labels' and 'confidences'
    within a specified 'box_cutoff' region of the data.

    Args:
        data: A tuple containing (root_id, features_dir, labels_dir, box_cutoff).

    Returns:
        A tuple containing (root_id, boolean_result), where boolean_result is True
        if 'no error in box' (meaning no `(1 - label) * confidence` is non-zero)
        and False otherwise.
    """
    root, features_dir, labels_dir, box_cutoff = data
    feature_path =  f'{features_dir}{root}.hdf5'
    labels_path = f'{labels_dir}{root}.hdf5'
    with h5py.File(feature_path, 'r') as feat_f, h5py.File(labels_path, 'r') as labels_f:
        labels = labels_f['labels'][:]
        confidences = labels_f['confidences'][:]
        rank = feat_f['rank'][:]
        size = len(labels)
        if size > box_cutoff:
            indices = np.where(rank < box_cutoff)[0]
            labels = labels[indices]
            confidences = confidences[indices]

        if np.any((1 - labels) * confidences): 
            return root, True
        else:
            return root, False

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time # Calculate elapsed time
    print(f"\nProcess completed in {elapsed_time:.2f} seconds.")
    
