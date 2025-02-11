from auto_proof.code.pre import data_utils

from caveclient import CAVEclient
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import multiprocessing
from datetime import timedelta


'''
Converts the raw edit history into usable dictionaries containing relevent information. 
Creates the intitial root list.
'''
# TODO: Currently won't work because the directories don't already exist in a barebones run
# so include that in Readme instructions
def process_raw_edits(config):
    client, _, mat_version = data_utils.create_client()
    data_directory = config['data']['data_dir']

    # TODO: If I already have the end dicts and root list then skip this entire method.
    # This should either go here or in pre_process_main. 
    # Seems like only the root list and root id to rep coords matter.

    og_df_path = f'{data_directory}240927/minnie_splitlog_240927.feather'
    og_edit_path = f'{data_directory}240927/minnie_splitlog_240927.npy'

    if not (os.path.exists(og_df_path) and os.path.exists(og_edit_path)):
        print("Error: initial starting df and edits doesn't exist")

    pruned_df_path = f'{data_directory}splitlog_{mat_version}/splitlog_{mat_version}.feather'    
    if not (os.path.exists(pruned_df_path)):
        print("saving pruned df")
         # [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
        og_df = pd.read_feather(og_df_path)
        save_pruned_df(client, mat_version, og_df, pruned_df_path)
        print("done saving pruned df")
    
    pruned_edit_path = f'{data_directory}splitlog_{mat_version}/splitlog_{mat_version}.npy'
    if not (os.path.exists(pruned_edit_path)):
        print("saving pruned edits")
        # [operation_id, sink supervoxel id, source_supervoxel id]
        og_edits = np.load(og_edit_path)
        pruned_df = pd.read_feather(pruned_df_path)
        save_pruned_edits(og_edits, pruned_df, pruned_edit_path)
        print("done saving pruned edits")

    op_to_svs_path = f'{data_directory}dicts/operation_to_supervoxels_{mat_version}.pkl'
    if not (os.path.exists(op_to_svs_path)):
        print("saving op_to_svs")
        pruned_edits = np.load(pruned_edit_path)
        save_operation_to_svs(pruned_edits, op_to_svs_path)
        print("done saving op_to_svs")

    op_to_pre_edit_dates_path = f'{data_directory}dicts/operation_to_pre_edit_dates_{mat_version}.pkl'
    if not (os.path.exists(op_to_pre_edit_dates_path)):
        print("saving op_to_pre_edit_dates")
        pruned_df = pd.read_feather(pruned_df_path)
        save_operation_to_pre_edit_dates(pruned_df, op_to_pre_edit_dates_path)
        print("done saving op_to_pre_edit_dates")
    
    op_to_rep_coords_path = f'{data_directory}dicts/operation_to_rep_coords_{mat_version}.pkl'
    if not (os.path.exists(op_to_rep_coords_path)):
        print("saving op_to_rep_coords")
        pruned_df = pd.read_feather(pruned_df_path)
        save_operation_to_rep_coords(pruned_df, op_to_rep_coords_path)
        print("done saving op_to_rep_coords")

    op_to_pre_edit_roots_path = f'{data_directory}dicts/operation_to_pre_edit_roots_{mat_version}.pkl'
    if not os.path.exists(op_to_pre_edit_roots_path):
        print("loading op_to_svs")
        op_to_svs = data_utils.load_pickle_dict(op_to_svs_path)
        print("loading op_to_pre_edit_dates")
        op_to_pre_edit_dates = data_utils.load_pickle_dict(op_to_pre_edit_dates_path)
        num_processes = config['loader']['num_workers']
        print("num processes", num_processes)
        print("saving op_to_pre_edit_roots")
        save_operation_to_pre_edit_roots(client, op_to_svs, op_to_pre_edit_dates, num_processes, op_to_pre_edit_roots_path)
        print("done saving op_to_pre_edit_roots")

    root_ids_txt_path = f'{data_directory}root_ids/pre_edit_roots_list_{mat_version}.txt'
    if not (os.path.exists(root_ids_txt_path)):
        print("saving roots txt file")
        op_to_pre_edit_roots = data_utils.load_pickle_dict(op_to_pre_edit_roots_path)
        op_to_pre_edit_roots_list = list(op_to_pre_edit_roots.values())
        data_utils.save_txt(root_ids_txt_path, op_to_pre_edit_roots_list)

    root_id_to_rep_coords_path = f'{data_directory}/root_id_to_rep_coords_{mat_version}.pkl'
    if not (os.path.exists(root_id_to_rep_coords_path)):
        print("saving root_id_to_rep_coords")
        op_to_rep_coords = data_utils.load_pickle_dict(op_to_rep_coords_path)
        op_to_pre_edit_roots = data_utils.load_pickle_dict(op_to_pre_edit_roots_path)
        save_root_id_to_rep_coords(op_to_rep_coords, op_to_pre_edit_roots, root_id_to_rep_coords_path)

'''
Prunes the original df to only contain operation ids that lie before and at materialization timestamp.
Inputs:
    client: CAVE client
    mat_version: materialization version
    df: Date to timestamp dataframe in format
        [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
    save_path: data path to save the file
'''
def save_pruned_df(client, mat_version, df, save_path):
    mat_timestamp = client.materialize.get_timestamp(version=mat_version)
    pruned_df = df.drop(df[df['date'] > mat_timestamp].index)
    pruned_df.to_feather(save_path)

'''
Prunes the original edit list to only contain operation ids from the input dataframe.
Inputs:
    edits: Edit history in the format [operation_id, sink supervoxel id, source_supervoxel id]
    df: Date to timestamp dataframe in format
        [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
    save_path: data path to save the file
'''
def save_pruned_edits(edits, df, save_path):
    prune_indices = []
    op_ids = set(df['operation_id'])
    for index, edit in enumerate(edits):
        if edit[0] in op_ids:
            prune_indices.append(index)
    pruned_edits = edits[prune_indices]
    np.save(save_path, pruned_edits)

'''
Creates a map from operation ids to supervoxels included in the split operations and saves it
Inputs:
    edits: Edit history in the format [operation_id, sink supervoxel id, source_supervoxel id]
    save_path: data path to save the file
'''
def save_operation_to_svs(edits, save_path):
    op_to_svs = defaultdict(set)
    for edit in edits:
        op_to_svs[edit[0]].add(edit[1])
        op_to_svs[edit[0]].add(edit[2])
    data_utils.save_pickle_dict(save_path, op_to_svs)

'''
Creates a map from operation ids to pre-edit dates 1 ms before edit time and saves it
Inputs:
    df: Date to timestamp dataframe in format
        [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
    save_path: data path to save the files
'''
def save_operation_to_pre_edit_dates(df, save_path):
    op_to_pre_edit_dates = {}
    one_ms = timedelta(milliseconds=1)
    for op_id in df['operation_id']:
        date = pd.to_datetime(df.loc[df['operation_id'] == op_id, 'date'].values[0])
        pre_edit_date = date - one_ms
        op_to_pre_edit_dates[op_id] = pre_edit_date
    data_utils.save_pickle_dict(save_path, op_to_pre_edit_dates)

'''
Creates a map from operation ids to representative coordinates.
Representative coords represent the x,y,z coordinate in nanometers
the point within the object that is designed to close to the "center" of the object.
Inputs:
    df: Date to timestamp dataframe in format
        [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
    save_path: data path to save the files
'''
def save_operation_to_rep_coords(df, save_path):
    op_to_rep_coords = {}
    
    for op_id in df['operation_id']:
        coord = np.zeros(3)
        coord[0] = df.loc[df['operation_id'] == op_id, 'mean_coord_x'].values[0] * 8
        coord[1] = df.loc[df['operation_id'] == op_id, 'mean_coord_y'].values[0] * 8
        coord[2] = df.loc[df['operation_id'] == op_id, 'mean_coord_z'].values[0] * 40
        op_to_rep_coords[op_id] = coord
    data_utils.save_pickle_dict(save_path, op_to_rep_coords)


'''
Creates a map from operation ids to pre-edit root ids 1 ms before edit time and saves it
Inputs:
    client: CAVE client
    op_to_svs: dict reprsesenting operation id to set of involved supervoxels
    op_to_pre_edit_dates: dict representing operation id to 1 ms pre edit date
    num_processes: number of processes to use for multiprocessing
    save_path: data path to save the file
'''
def save_operation_to_pre_edit_roots(client, op_to_svs, op_to_pre_edit_dates, num_processes, save_path):

    # Make ops smaller for testing
    ops = np.array(list(op_to_pre_edit_dates.keys()))
    # ops = np.array(list(op_to_pre_edit_dates.keys()))[:5000]
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

'''
Individual operation per process for save_operation_to_pre_edit_roots
'''
def __process_operation__(data):
    op_id, date, supervoxel, client = data
    root_id = client.chunkedgraph.get_root_id(supervoxel, date)
    return op_id, root_id 

'''
Creates a map from root ids to rep coords and saves it
Inputs:
    op_to_rep_coords: dict representing operation id to representative coordinate of edit
    op_to_pre_edit_roots: dict representing operation id to root id 1 ms before
    save_path: data path to save the file
'''
def save_root_id_to_rep_coords(op_to_rep_coords, op_to_pre_edit_roots, save_path):
    root_id_to_rep_coords_dict = {}
    for op in op_to_rep_coords:
        root_id_to_rep_coords_dict[op_to_pre_edit_roots[op]] = op_to_rep_coords[op]
    data_utils.save_pickle_dict(save_path, root_id_to_rep_coords_dict)


if __name__ == "__main__":
    config = data_utils.get_config()
    process_raw_edits(config)