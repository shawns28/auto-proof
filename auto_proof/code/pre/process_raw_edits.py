from auto_proof.code.pre import data_utils

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import multiprocessing
from datetime import timedelta
import argparse

def process_raw_edits(data_config):
    """Converts the raw edit history into usable dictionaries containing relevent information. 
    TODO: Fill in

    Creates initial root list and root to representative coordinate dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_processes", help="num processes")
    args = parser.parse_args()
    if args.num_processes:
        data_config['multiprocessing']['num_processes'] = int(args.num_processes)
    
    client = data_utils.create_client(data_config)
    mat_version_start = data_config['client']['mat_version_start']
    mat_version_end = data_config['client']['mat_version_end']
    data_dir = data_config['paths']['data_dir']

    splitlog_dir = f'{data_dir}splitlog_{mat_version_start}_{mat_version_end}/'
    if not os.path.exists(splitlog_dir):
        os.makedirs(splitlog_dir)
    dicts_dir = f'{data_dir}dicts_{mat_version_start}_{mat_version_end}/'
    if not os.path.exists(dicts_dir):
        os.makedirs(dicts_dir)
    roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'
    if not os.path.exists(dicts_dir):
        os.makedirs(dicts_dir)

    root_ids_txt_path = f'{roots_dir}pre_edit_roots_list.txt'
    root_id_to_rep_coords_path = f'{dicts_dir}root_id_to_rep_coords.pkl'
    if os.path.exists(root_ids_txt_path) and os.path.exists(root_id_to_rep_coords_path):
        print("Already have the root id and rep list")
        return 

    raw_df_path = data_config['raw_edits']['raw_df']
    raw_edit_path = data_config['raw_edits']['raw_edit']

    if not (os.path.exists(raw_df_path) and os.path.exists(raw_edit_path)):
        print("Error: initial starting df and edits doesn't exist")
        return

    pruned_df_path = f'{splitlog_dir}splitlog.feather'    
    if not (os.path.exists(pruned_df_path)):
        print("saving pruned df")
         # [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
        og_df = pd.read_feather(raw_df_path)
        save_pruned_df(client, mat_version_start, mat_version_end, og_df, pruned_df_path)
        print("done saving pruned df")
    
    pruned_edit_path = f'{splitlog_dir}splitlog.npy'
    if not (os.path.exists(pruned_edit_path)):
        print("saving pruned edits")
        # [operation_id, sink supervoxel id, source_supervoxel id] Should be in comments
        og_edits = np.load(raw_edit_path)
        pruned_df = pd.read_feather(pruned_df_path)
        save_pruned_edits(og_edits, pruned_df, pruned_edit_path)
        print("done saving pruned edits")

    op_to_svs_path = f'{dicts_dir}operation_to_supervoxels.pkl'
    if not (os.path.exists(op_to_svs_path)):
        print("saving op_to_svs")
        pruned_edits = np.load(pruned_edit_path)
        save_operation_to_svs(pruned_edits, op_to_svs_path)
        print("done saving op_to_svs")

    op_to_pre_edit_dates_path = f'{dicts_dir}operation_to_pre_edit_dates.pkl'
    if not (os.path.exists(op_to_pre_edit_dates_path)):
        print("saving op_to_pre_edit_dates")
        pruned_df = pd.read_feather(pruned_df_path)
        save_operation_to_pre_edit_dates(pruned_df, op_to_pre_edit_dates_path)
        print("done saving op_to_pre_edit_dates")
    
    op_to_rep_coords_path = f'{dicts_dir}operation_to_rep_coords.pkl'
    if not (os.path.exists(op_to_rep_coords_path)):
        print("saving op_to_rep_coords")
        pruned_df = pd.read_feather(pruned_df_path)
        x_res = data_config['raw_edits']['x_res']
        y_res = data_config['raw_edits']['y_res']
        z_res = data_config['raw_edits']['z_res']
        save_operation_to_rep_coords(pruned_df, x_res, y_res, z_res, op_to_rep_coords_path)
        print("done saving op_to_rep_coords")

    op_to_pre_edit_roots_path = f'{dicts_dir}operation_to_pre_edit_roots.pkl'
    if not os.path.exists(op_to_pre_edit_roots_path):
        print("loading op_to_svs")
        op_to_svs = data_utils.load_pickle_dict(op_to_svs_path)
        print("loading op_to_pre_edit_dates")
        op_to_pre_edit_dates = data_utils.load_pickle_dict(op_to_pre_edit_dates_path)
        num_processes = data_config['multiprocessing']['num_processes']
        print("num processes", num_processes)
        print("saving op_to_pre_edit_roots")
        save_operation_to_pre_edit_roots(client, op_to_svs, op_to_pre_edit_dates, num_processes, op_to_pre_edit_roots_path)
        print("done saving op_to_pre_edit_roots")

    if not (os.path.exists(root_ids_txt_path)):
        print("saving roots txt file")
        op_to_pre_edit_roots = data_utils.load_pickle_dict(op_to_pre_edit_roots_path)
        op_to_pre_edit_roots_list = list(op_to_pre_edit_roots.values())
        data_utils.save_txt(root_ids_txt_path, op_to_pre_edit_roots_list)

    if not (os.path.exists(root_id_to_rep_coords_path)):
        print("saving root_id_to_rep_coords")
        op_to_rep_coords = data_utils.load_pickle_dict(op_to_rep_coords_path)
        op_to_pre_edit_roots = data_utils.load_pickle_dict(op_to_pre_edit_roots_path)
        save_root_id_to_rep_coords(op_to_rep_coords, op_to_pre_edit_roots, root_id_to_rep_coords_path)

def save_pruned_df(client, mat_version_start, mat_version_end, df, save_path):
    """Prunes the original df to only contain operation ids that lie before and at materialization timestamp.
    
    Args:
        client: CAVE client
        mat_version_start: materialization version to start pruning from inclusive
        mat_version_end: materialization version to end edits inclusive
        df: Date to timestamp dataframe in format
            [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
        save_path: data path to save the file
    """
    df = df.sort_values(by=['date'])
    print(df.head(5))
    print(df.tail(5))
    mat_end_timestamp = client.materialize.get_timestamp(version=mat_version_end)
    print("mat end timestamp", mat_end_timestamp)
    df = df.drop(df[df['date'] > mat_end_timestamp].index)
    print(df.head(5))
    print(df.tail(5))
    mat_start_timestamp = client.materialize.get_timestamp(version=mat_version_start)
    one_ms = timedelta(milliseconds=1)
    mat_start_timestamp = mat_start_timestamp + one_ms
    print("mat_start_timestamp + 1", mat_start_timestamp)
    df = df.drop(df[df['date'] < mat_start_timestamp].index)
    print(df.head(5))
    print(df.tail(5))
    df.to_feather(save_path)

def save_pruned_edits(edits, df, save_path):
    """Prunes the original edit list to only contain operation ids from the input dataframe.

    Args:
        edits: Edit history in the format [operation_id, sink supervoxel id, source_supervoxel id]
        df: Date to timestamp dataframe in format
            [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
        save_path: data path to save the file
    """
    prune_indices = []
    op_ids = set(df['operation_id'])
    for index, edit in enumerate(edits):
        if edit[0] in op_ids:
            prune_indices.append(index)
    pruned_edits = edits[prune_indices]
    np.save(save_path, pruned_edits)

def save_operation_to_svs(edits, save_path):
    """Creates a map from operation ids to supervoxels included in the split operations and saves it
    
    Args:
        edits: Edit history in the format [operation_id, sink supervoxel id, source_supervoxel id]
        save_path: data path to save the file
    """
    op_to_svs = defaultdict(set)
    for edit in edits:
        op_to_svs[edit[0]].add(edit[1])
        op_to_svs[edit[0]].add(edit[2])
    data_utils.save_pickle_dict(save_path, op_to_svs)

def save_operation_to_pre_edit_dates(df, save_path):
    """
    Creates a map from operation ids to pre-edit dates 1 ms before edit time and saves it

    Args:
        df: Date to timestamp dataframe in format
            [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
        save_path: data path to save the files
    """
    op_to_pre_edit_dates = {}
    one_ms = timedelta(milliseconds=1)
    for op_id in df['operation_id']:
        date = pd.to_datetime(df.loc[df['operation_id'] == op_id, 'date'].values[0])
        pre_edit_date = date - one_ms
        op_to_pre_edit_dates[op_id] = pre_edit_date
    data_utils.save_pickle_dict(save_path, op_to_pre_edit_dates)

def save_operation_to_rep_coords(df, x_res, y_res, z_res, save_path):
    """
    Creates a map from operation ids to representative coordinates.
    Representative coords represent the x,y,z coordinate in nanometers
    the point within the object that is designed to close to the "center" of the object.

    Args:
        df: Date to timestamp dataframe in format
            [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
        save_path: data path to save the files
    """
    op_to_rep_coords = {}
    
    for op_id in df['operation_id']:
        coord = np.zeros(3)
        coord[0] = df.loc[df['operation_id'] == op_id, 'mean_coord_x'].values[0] * 8
        coord[1] = df.loc[df['operation_id'] == op_id, 'mean_coord_y'].values[0] * 8
        coord[2] = df.loc[df['operation_id'] == op_id, 'mean_coord_z'].values[0] * 40
        op_to_rep_coords[op_id] = coord
    data_utils.save_pickle_dict(save_path, op_to_rep_coords)

def __process_operation__(data):
    """Individual operation per process for save_operation_to_pre_edit_roots"""
    op_id, date, supervoxel, client = data
    root_id = client.chunkedgraph.get_root_id(supervoxel, date)
    return op_id, root_id 

def save_operation_to_pre_edit_roots(client, op_to_svs, op_to_pre_edit_dates, num_processes, save_path):
    """
    Creates a map from operation ids to pre-edit root ids 1 ms before edit time and saves it
    
    Args:
        client: CAVE client
        op_to_svs: dict reprsesenting operation id to set of involved supervoxels
        op_to_pre_edit_dates: dict representing operation id to 1 ms pre edit date
        num_processes: number of processes to use for multiprocessing
        save_path: data path to save the file
    """
    ops = np.array(list(op_to_pre_edit_dates.keys()))
    args_list = list([(ops[i], op_to_pre_edit_dates[ops[i]], next(iter(op_to_svs[ops[i]])), client) for i in range(len(ops))])
    np_args_list = np.asarray(args_list, dtype="object")
    manager = multiprocessing.Manager()
    op_to_pre_edit_roots = manager.dict()
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(ops)) as pbar:
        for op_id, root_id in pool.imap_unordered(__process_operation__, np_args_list):
            op_to_pre_edit_roots[op_id] = root_id
            pbar.update()

    op_to_pre_edit_roots_classic_dict = dict(op_to_pre_edit_roots)
    data_utils.save_pickle_dict(save_path, op_to_pre_edit_roots_classic_dict)

def save_root_id_to_rep_coords(op_to_rep_coords, op_to_pre_edit_roots, save_path):
    """Creates a map from root ids to rep coords and saves it

    Args:
        op_to_rep_coords: dict representing operation id to representative coordinate of edit
        op_to_pre_edit_roots: dict representing operation id to root id 1 ms before
        save_path: data path to save the file
    """
    root_id_to_rep_coords_dict = {}
    for op in op_to_rep_coords:
        root_id_to_rep_coords_dict[op_to_pre_edit_roots[op]] = op_to_rep_coords[op]
    data_utils.save_pickle_dict(save_path, root_id_to_rep_coords_dict)

if __name__ == "__main__":
    data_config = data_utils.get_data_config()
    process_raw_edits(data_config)