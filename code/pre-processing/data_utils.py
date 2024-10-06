import caveclient
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import multiprocessing
from tqdm import tqdm
import pickle
from datetime import timedelta

'''
Prunes the original df and edit list to only contain operation ids that lie before and at materialization timestamp.
Inputs:
    client: CAVE client
    mat_version: materialization version
    edits: Edit history in the format [operation_id, sink supervoxel id, source_supervoxel id]
    df: Date to timestamp dataframe in format
        [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
    save_directory: data path to save the files
Returns:
    Saves the pruned df and edit list as "save_directory"/minnie_splitlog_"matversion".feather / .npy
'''
def prune_edits_before_mat(client, mat_version, edits, df, save_directory):
    mat_timestamp = client.materialize.get_timestamp(version=mat_version)
    
    pruned_df = df.drop(df[df['date'] > mat_timestamp].index)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    pruned_df.to_feather(f"{save_directory}/splitlog_{mat_version}.feather")

    prune_indices = []
    op_ids = set(pruned_df['operation_id'])
    for index, edit in enumerate(edits):
        if edit[0] in op_ids:
            prune_indices.append(index)
    pruned_edits = edits[prune_indices]
    np.save(f"{save_directory}/splitlog_{mat_version}.npy", pruned_edits)

'''
Creates a map from operation ids to supervoxels included in the split operations
Inputs:
    client: CAVE client
    mat_version: materialization version
    edits: Edit history in the format [operation_id, sink supervoxel id, source_supervoxel id]
    save_directory: data path to save the files
    save: If we should save the resulting dict at 
        "save_directory"/operation_to_supervoxels_"mat_version".pkl
Returns:
    a dict representing the operation id to involved supervoxels
'''
def create_operation_to_svs(client, mat_version, edits, save_directory, save=False):
    op_to_svs = defaultdict(set)
    for edit in edits:
        op_to_svs[edit[0]].add(edit[1])
        op_to_svs[edit[0]].add(edit[2])
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if save:
        with open(f"{save_directory}/operation_to_supervoxels_{mat_version}.pkl", 'wb') as f:
            pickle.dump(op_to_svs, f)
    return op_to_svs

'''
Creates a map from operation ids to pre-edit dates 1 ms before edit time
Inputs:
    mat_version: materialization version
    df: Date to timestamp dataframe in format
        [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
    save_directory: data path to save the files
    save: If we should save the resulting dict at 
    "save_directory"/operation_to_pre_edit_dates_"mat_version".pkl
Returns:
    a dict representing the operation id to involved supervoxels
'''
def create_operation_to_pre_edit_dates(mat_version, df, save_directory, save=False):
    op_to_pre_edit_dates = {}
    one_ms = timedelta(milliseconds=1)
    for op_id in df['operation_id']:
        date = pd.to_datetime(df.loc[df['operation_id'] == op_id, 'date'].values[0])
        pre_edit_date = date - one_ms
        op_to_pre_edit_dates[op_id] = pre_edit_date
    if save:
        with open(f"{save_directory}/operation_to_pre_edit_dates_{mat_version}.pkl", 'wb') as f:
            pickle.dump(op_to_pre_edit_dates, f)
    return op_to_pre_edit_dates


'''
Creates a map from operation ids to pre-edit root ids 1 ms before edit time
Inputs:
    client: CAVE client
    mat_version: materialization version
    op_to_pre_edit_dates: dict representing operation id to 1 ms pre edit date
    op_to_svs: dict reprsesenting operation id to set of involved supervoxels
    num_processes: number of processes to use for multiprocessing
    save_directory: data path to save the files
    save: If we should save the resulting dict at 
    "save_directory"/operation_to_pre_edit_roots_"mat_version".pkl
Returns:
    a dict representing the operation id to root id
'''
def create_operation_to_pre_edit_roots(client, mat_version, op_to_pre_edit_dates, op_to_svs, num_processes, save_directory, save=False):
    def fetch_root_id(client, supervoxel, date):
        root_id = client.chunkedgraph.get_root_id(supervoxel, date)
        return root_id

    def process_operation(data):
        op_id, date, supervoxel, client = data
        root_id = fetch_root_id(client, supervoxel, date)
        return op_id, root_id 

    ops = np.array(list(op_to_pre_edit_dates.keys()))
    args_list = list([(ops[i], op_to_pre_edit_dates[ops[i]], op_to_svs[ops[i]].pop(), client) for i in range(len(ops))])
    np_args_list = np.asarray(args_list, dtype="object")
    manager = multiprocessing.Manager()
    op_to_pre_edit_roots = manager.dict()
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(ops)) as pbar:
        for op_id, root_id in pool.imap_unordered(process_operation, np_args_list):
            op_to_pre_edit_roots[op_id] = root_id
            pbar.update()
    if save:
        with open(f"{save_directory}/operation_to_pre_edit_roots_{mat_version}.pkl", 'wb') as f:
            pickle.dump(op_to_pre_edit_roots, f)
    return op_to_pre_edit_roots

