from auto_proof.code.pre import data_utils

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import multiprocessing
from datetime import timedelta
import argparse
import time

def main():
    """Converts the raw edit history into usable dictionaries containing relevant information.

    This function orchestrates the entire preprocessing pipeline for raw edit data.
    It performs the following steps:
    1. Loads configuration settings.
    2. Sets up necessary directories for storing processed data.
    3. Prunes the initial DataFrame and edit array based on materialization timestamps.
    4. Creates and saves various mappings:
        - Operation ID to involved supervoxels.
        - Operation ID to pre-edit timestamps.
        - Operation ID to representative edit coordinates.
        - Operation ID to pre-edit root IDs (using multiprocessing).
    5. Generates an initial list of root IDs and a mapping from root ID to representative edit coordinates.
    """
    data_config = data_utils.get_config('data')
    client_config = data_utils.get_config('client')
    parser = argparse.ArgumentParser(
        description="Process raw edit history into structured data."
    )
    parser.add_argument("-n", "--num_processes", type=int,
                        help="Number of processes to use for multiprocessing.")
    args = parser.parse_args()

    if args.num_processes:
        data_config['multiprocessing']['num_processes'] = int(args.num_processes)

    client, _, mat_version_start, mat_version_end = data_utils.create_client(client_config)
    data_dir = data_config['data_dir']

    # Define and create directories for output files.
    splitlog_dir = f'{data_dir}splitlog_{mat_version_start}_{mat_version_end}/'
    os.makedirs(splitlog_dir, exist_ok=True)

    dicts_dir = f'{data_dir}dicts_{mat_version_start}_{mat_version_end}/'
    os.makedirs(dicts_dir, exist_ok=True)

    roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'
    os.makedirs(roots_dir, exist_ok=True)

    # For roots between specific materialization versions, used for SegCLR
    intermediate_mat_version_end = data_config['raw_edits']['intermediate_version']
    roots_between_versions_txt_path = f'{roots_dir}roots_{mat_version_start}_{intermediate_mat_version_end}.txt'

    root_ids_txt_path = f'{roots_dir}{data_config['raw_edits']['post_raw_edit_roots']}'
    root_id_to_rep_coords_path = f'{dicts_dir}{data_config['raw_edits']['root_to_rep']}'

    # Skip processing if final output files already exist.
    if (os.path.exists(root_ids_txt_path) and os.path.exists(root_id_to_rep_coords_path)) and os.path.exists(roots_between_versions_txt_path):
        print("Already have the total/intermediary root lists and representative coordinate dict. Skipping processing.")
        return

    raw_df_path = data_config['raw_edits']['raw_df']
    raw_edit_path = data_config['raw_edits']['raw_edit']

    # Ensure input raw data files exist.
    if not (os.path.exists(raw_df_path) and os.path.exists(raw_edit_path)):
        print("Error: Initial raw DataFrame and edits file do not exist. Exiting.")
        return

    # Prune and save the DataFrame containing split log metadata.
    pruned_df_path = f'{splitlog_dir}splitlog.feather'
    if not os.path.exists(pruned_df_path):
        print("Saving pruned DataFrame...")
        # `original_df` format: [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
        original_df = pd.read_feather(raw_df_path)
        save_pruned_df(client, mat_version_start, mat_version_end, original_df, pruned_df_path)
        print("Done saving pruned DataFrame.")

    # Prune and save the raw edits (supervoxel merges/splits).
    pruned_edit_path = f'{splitlog_dir}splitlog.npy'
    if not os.path.exists(pruned_edit_path):
        print("Saving pruned edits...")
        # `original_edits` format: [operation_id, sink_supervoxel_id, source_supervoxel_id]
        original_edits = np.load(raw_edit_path)
        pruned_df = pd.read_feather(pruned_df_path)
        save_pruned_edits(original_edits, pruned_df, pruned_edit_path)
        print("Done saving pruned edits.")

    # Create and save a mapping from operation IDs to involved supervoxels.
    op_to_svs_path = f'{dicts_dir}operation_to_supervoxels.pkl'
    if not os.path.exists(op_to_svs_path):
        print("Saving operation to supervoxels mapping...")
        pruned_edits = np.load(pruned_edit_path)
        save_operation_to_svs(pruned_edits, op_to_svs_path)
        print("Done saving operation to supervoxels mapping.")

    # Create and save a mapping from operation IDs to their pre-edit dates.
    op_to_pre_edit_dates_path = f'{dicts_dir}operation_to_pre_edit_dates.pkl'
    if not os.path.exists(op_to_pre_edit_dates_path):
        print("Saving operation to pre-edit dates mapping...")
        pruned_df = pd.read_feather(pruned_df_path)
        save_operation_to_pre_edit_dates(pruned_df, op_to_pre_edit_dates_path)
        print("Done saving operation to pre-edit dates mapping.")

    # Create and save a mapping from operation IDs to their representative coordinates.
    op_to_rep_coords_path = f'{dicts_dir}operation_to_rep_coords.pkl'
    if not os.path.exists(op_to_rep_coords_path):
        print("Saving operation to representative coordinates mapping...")
        pruned_df = pd.read_feather(pruned_df_path)
        resolution = data_config['segmentation']['resolution']
        save_operation_to_rep_coords(pruned_df, resolution, op_to_rep_coords_path)
        print("Done saving operation to representative coordinates mapping.")

    # Create and save a mapping from operation IDs to pre-edit root IDs.
    op_to_pre_edit_roots_path = f'{dicts_dir}operation_to_pre_edit_roots.pkl'
    if not os.path.exists(op_to_pre_edit_roots_path):
        print("Loading operation to supervoxels mapping...")
        op_to_svs = data_utils.load_pickle_dict(op_to_svs_path)
        print("Loading operation to pre-edit dates mapping...")
        op_to_pre_edit_dates = data_utils.load_pickle_dict(op_to_pre_edit_dates_path)
        num_processes = data_config['multiprocessing']['num_processes']
        print(f"Using {num_processes} processes for root ID lookup.")
        print("Saving operation to pre-edit roots mapping...")
        save_operation_to_pre_edit_roots(
            client, op_to_svs, op_to_pre_edit_dates, num_processes, op_to_pre_edit_roots_path
        )
        print("Done saving operation to pre-edit roots mapping.")

    # Create and save a text file containing all initial root IDs.
    if not os.path.exists(root_ids_txt_path):
        print("Saving root IDs to text file...")
        op_to_pre_edit_roots = data_utils.load_pickle_dict(op_to_pre_edit_roots_path)
        op_to_pre_edit_roots_list = list(op_to_pre_edit_roots.values())
        data_utils.save_txt(root_ids_txt_path, op_to_pre_edit_roots_list)
        print("Done saving root IDs to text file.")

    # Create and save a mapping from root IDs to representative coordinates.
    if not os.path.exists(root_id_to_rep_coords_path):
        print("Saving root ID to representative coordinates mapping...")
        op_to_rep_coords = data_utils.load_pickle_dict(op_to_rep_coords_path)
        op_to_pre_edit_roots = data_utils.load_pickle_dict(op_to_pre_edit_roots_path)
        save_root_id_to_rep_coords(
            op_to_rep_coords, op_to_pre_edit_roots, root_id_to_rep_coords_path
        )
        print("Done saving root ID to representative coordinates mapping.")

    # Create and save a text file for roots between mat_version_start and intermediate_mat_version_end
    if not os.path.exists(roots_between_versions_txt_path):
        print(f"Saving root IDs between materialization versions {mat_version_start} and {intermediate_mat_version_end}...")
        op_to_pre_edit_roots = data_utils.load_pickle_dict(op_to_pre_edit_roots_path)
        op_to_pre_edit_dates = data_utils.load_pickle_dict(op_to_pre_edit_dates_path)
        save_roots_between_mat_versions(
            client, op_to_pre_edit_roots, op_to_pre_edit_dates,
            mat_version_start, intermediate_mat_version_end,
            roots_between_versions_txt_path
        )
        print(f"Done saving root IDs between materialization versions {mat_version_start} and {intermediate_mat_version_end}.")

def save_pruned_df(client, mat_version_start: int, mat_version_end: int, df: pd.DataFrame, save_path: str):
    """Prunes the input DataFrame to include only operations within a specified
    materialization version range.

    Operations with timestamps outside the [mat_version_start, mat_version_end]
    range (inclusive for the end, exclusive for the start after adding 1ms) are dropped.

    Args:
        client: The CAVE client object, used to get materialization timestamps.
        mat_version_start: The starting materialization version (exclusive).
        mat_version_end: The ending materialization version (inclusive).
        df: The input Pandas DataFrame with columns including 'date' (timestamps)
            and 'operation_id'.
        save_path: The file path to save the pruned DataFrame in Feather format.
    """
    df = df.sort_values(by=['date'])
    print("Original DataFrame head (first 5 rows):\n", df.head(5))
    print("Original DataFrame tail (last 5 rows):\n", df.tail(5))

    # Get the timestamp for the end materialization version.
    mat_end_timestamp = client.materialize.get_timestamp(version=mat_version_end)
    print(f"Materialization end timestamp: {mat_end_timestamp}")

    # Drop rows where 'date' is after the end materialization timestamp.
    df = df.drop(df[df['date'] > mat_end_timestamp].index)
    print("DataFrame after pruning by end timestamp - head:\n", df.head(5))
    print("DataFrame after pruning by end timestamp - tail:\n", df.tail(5))

    # Get the timestamp for the start materialization version.
    mat_start_timestamp = client.materialize.get_timestamp(version=mat_version_start)
    # Add 1 millisecond to `mat_start_timestamp` to make the start exclusive.
    one_ms = timedelta(milliseconds=1)
    mat_start_timestamp = mat_start_timestamp + one_ms
    print(f"Materialization start timestamp + 1ms: {mat_start_timestamp}")

    # Drop rows where 'date' is before the start materialization timestamp.
    df = df.drop(df[df['date'] < mat_start_timestamp].index)
    print("DataFrame after pruning by start timestamp - head:\n", df.head(5))
    print("DataFrame after pruning by start timestamp - tail:\n", df.tail(5))

    df.to_feather(save_path)

def save_pruned_edits(edits: np.ndarray, df: pd.DataFrame, save_path: str):
    """Prunes the raw edit array to include only edits whose operation IDs are present
    in the provided DataFrame.

    Args:
        edits: A NumPy array representing the raw edit history.
               Expected format: `[operation_id, sink_supervoxel_id, source_supervoxel_id]`.
        df: A Pandas DataFrame containing a 'operation_id' column, indicating
            the operations to keep.
        save_path: The file path to save the pruned edits as a NumPy array.
    """
    prune_indices = []
    op_ids = set(df['operation_id'])  # Use a set for efficient lookup.
    for index, edit in enumerate(edits):
        if edit[0] in op_ids:
            prune_indices.append(index)

    pruned_edits = edits[prune_indices]
    np.save(save_path, pruned_edits)


def save_operation_to_svs(edits: np.ndarray, save_path: str):
    """Creates a dictionary mapping each operation ID to a set of supervoxel IDs
    involved in that operation.

    Args:
        edits: A NumPy array representing the edit history.
               Expected format: `[operation_id, sink_supervoxel_id, source_supervoxel_id]`.
        save_path: The file path to save the resulting dictionary as a pickle file.
    """
    op_to_svs = defaultdict(set)
    for edit in edits:
        op_to_svs[edit[0]].add(edit[1])  # Add sink supervoxel.
        op_to_svs[edit[0]].add(edit[2])  # Add source supervoxel.
    data_utils.save_pickle_dict(save_path, op_to_svs)


def save_operation_to_pre_edit_dates(df: pd.DataFrame, save_path: str):
    """Creates a dictionary mapping each operation ID to a timestamp
    representing 1 millisecond *before* the operation's recorded edit time.

    Args:
        df: A Pandas DataFrame with 'operation_id' and 'date' columns.
            'date' should contain timestamps.
        save_path: The file path to save the resulting dictionary as a pickle file.
    """
    op_to_pre_edit_dates = {}
    one_ms = timedelta(milliseconds=1)
    for op_id in df['operation_id']:
        date = pd.to_datetime(df.loc[df['operation_id'] == op_id, 'date'].values[0])
        pre_edit_date = date - one_ms
        op_to_pre_edit_dates[op_id] = pre_edit_date
    data_utils.save_pickle_dict(save_path, op_to_pre_edit_dates)

def save_operation_to_rep_coords(df: pd.DataFrame, resolution: tuple[float, float, float], save_path: str):
    """Creates a dictionary mapping each operation ID to its representative
    edit coordinates in nanometers.

    Representative coordinates are derived from mean coordinates in the DataFrame,
    scaled by the provided resolution.

    Args:
        df: A Pandas DataFrame with 'operation_id', 'mean_coord_x',
            'mean_coord_y', and 'mean_coord_z' columns.
        resolution: A tuple or list `(res_x, res_y, res_z)` representing the
                    resolution in nanometers per unit for each dimension.
        save_path: The file path to save the resulting dictionary as a pickle file.
    """
    op_to_rep_coords = {}

    for op_id in df['operation_id'].unique():
        coord = np.zeros(3)
        # Extract mean coordinates and scale by resolution to get nanometer values.
        coord[0] = df.loc[df['operation_id'] == op_id, 'mean_coord_x'].values[0] * resolution[0]
        coord[1] = df.loc[df['operation_id'] == op_id, 'mean_coord_y'].values[0] * resolution[1]
        coord[2] = df.loc[df['operation_id'] == op_id, 'mean_coord_z'].values[0] * resolution[2]
        op_to_rep_coords[op_id] = coord
    data_utils.save_pickle_dict(save_path, op_to_rep_coords)


def __process_operation__(data: tuple) -> tuple[int, int]:
    """Helper function to fetch the root ID for a given supervoxel at a specific date.

    This function is designed to be used with multiprocessing to parallelize
    root ID lookups from the CAVE client.

    Args:
        data: A tuple containing (operation_id, timestamp, supervoxel_id, CAVE_client).

    Returns:
        A tuple containing (operation_id, root_id) for the given supervoxel
        at the specified timestamp.
    """
    op_id, date, supervoxel, client = data
    root_id = client.chunkedgraph.get_root_id(supervoxel, date)
    return op_id, root_id


def save_operation_to_pre_edit_roots(
    client,
    op_to_svs: dict[int, set[int]],
    op_to_pre_edit_dates: dict[int, pd.Timestamp],
    num_processes: int,
    save_path: str
):
    """Creates a dictionary mapping each operation ID to the root ID of an involved
    supervoxel *before* the edit was applied.

    This involves querying the CAVE client for the root ID of one of the supervoxels
    involved in the operation, using the pre-edit timestamp. Multiprocessing is
    used to speed up these queries.

    Args:
        client: The CAVE client object.
        op_to_svs: A dictionary mapping operation IDs to sets of involved supervoxel IDs.
        op_to_pre_edit_dates: A dictionary mapping operation IDs to their pre-edit timestamps.
        num_processes: The number of parallel processes to use for root ID lookups.
        save_path: The file path to save the resulting dictionary as a pickle file.
    """
    ops = np.array(list(op_to_pre_edit_dates.keys()))
    # Prepare arguments for multiprocessing: (op_id, date, first_supervoxel, client)
    # It assumes that taking the first supervoxel from `op_to_svs[ops[i]]` is sufficient,
    # implying all supervoxels in a given operation belonged to the same root pre-edit.
    args_list = [
        (op_id, op_to_pre_edit_dates[op_id], next(iter(op_to_svs[op_id])), client)
        for op_id in ops
    ]
    # Using `np.asarray(..., dtype="object")` is important when the array elements
    # are Python objects (like lists, dictionaries, or custom classes).
    np_args_list = np.asarray(args_list, dtype="object")

    # Use a multiprocessing.Manager.dict() to share the dictionary across processes.
    manager = multiprocessing.Manager()
    op_to_pre_edit_roots = manager.dict()

    with multiprocessing.Pool(processes=num_processes) as pool, \
         tqdm(total=len(ops), desc="Fetching pre-edit root IDs") as pbar:
        # `imap_unordered` is used for faster results as soon as they are ready.
        for op_id, root_id in pool.imap_unordered(__process_operation__, np_args_list):
            op_to_pre_edit_roots[op_id] = root_id
            pbar.update()

    # Convert ManagerDict back to a regular dict before saving.
    op_to_pre_edit_roots_classic_dict = dict(op_to_pre_edit_roots)
    data_utils.save_pickle_dict(save_path, op_to_pre_edit_roots_classic_dict)

def save_root_id_to_rep_coords(
    op_to_rep_coords: dict[int, np.ndarray],
    op_to_pre_edit_roots: dict[int, int],
    save_path: str
):
    """Creates a dictionary mapping each pre-edit root ID to its representative
    coordinate.

    This is achieved by using the `op_to_pre_edit_roots` mapping to link
    operation IDs (for which representative coordinates are known) to their
    corresponding root IDs.

    Args:
        op_to_rep_coords: A dictionary mapping operation IDs to their representative
                          coordinates.
        op_to_pre_edit_roots: A dictionary mapping operation IDs to their
                              pre-edit root IDs.
        save_path: The file path to save the resulting dictionary as a pickle file.
    """
    root_id_to_rep_coords_dict = {}
    for op_id, rep_coord in op_to_rep_coords.items():
        root_id_to_rep_coords_dict[op_to_pre_edit_roots[op_id]] = rep_coord
    data_utils.save_pickle_dict(save_path, root_id_to_rep_coords_dict)

def save_roots_between_mat_versions(
    client,
    op_to_pre_edit_roots: dict[int, int],
    op_to_pre_edit_dates: dict[int, pd.Timestamp],
    mat_version_start: int,
    intermediate_mat_version_end: int,
    save_path: str
):
    """Identifies and saves a list of unique root IDs corresponding to operations
    that occurred between a specified start materialization version (exclusive)
    and an intermediate end materialization version (inclusive).

    Args:
        client: The CAVE client object, used to get materialization timestamps.
        op_to_pre_edit_roots: A dictionary mapping operation IDs to their pre-edit root IDs.
        op_to_pre_edit_dates: A dictionary mapping operation IDs to their pre-edit timestamps.
        mat_version_start: The starting materialization version for the range (exclusive).
        intermediate_mat_version_end: The ending materialization version for the range (inclusive).
        save_path: The file path to save the unique root IDs as a text file.
    """
    roots_in_range = set()

    # Get timestamps for the specified materialization versions
    start_timestamp = client.materialize.get_timestamp(version=mat_version_start)
    # Add 1 millisecond to `start_timestamp` to make the start exclusive for operations.
    start_timestamp += timedelta(milliseconds=1)
    end_timestamp = client.materialize.get_timestamp(version=intermediate_mat_version_end)

    print(f"Filtering roots between {start_timestamp} (exclusive) and {end_timestamp} (inclusive).")

    for op_id, root_id in tqdm(op_to_pre_edit_roots.items(), desc="Filtering roots by date range"):
        pre_edit_date = op_to_pre_edit_dates.get(op_id).tz_localize('UTC')

        if start_timestamp < pre_edit_date <= end_timestamp:
            roots_in_range.add(root_id)

    data_utils.save_txt(save_path, list(roots_in_range))

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nProcess completed in {elapsed_time:.2f} seconds.")