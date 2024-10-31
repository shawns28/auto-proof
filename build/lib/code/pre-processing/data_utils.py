from caveclient import CAVEclient
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import multiprocessing
from tqdm import tqdm
import pickle
from datetime import timedelta
import json
import h5py

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
    save_pickle_dict(save_path, op_to_svs)

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
    save_pickle_dict(save_path, op_to_pre_edit_dates)

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
    save_pickle_dict(save_path, op_to_rep_coords)


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
    save_pickle_dict(save_path, op_to_pre_edit_roots_classic_dict)


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
    save_pickle_dict(save_path, root_id_to_rep_coords_dict)
'''
Private method
'''
def __process_operation__(data):
    op_id, date, supervoxel, client = data
    root_id = client.chunkedgraph.get_root_id(supervoxel, date)
    return op_id, root_id 


'''
Saves dictionary as pickle file at filepath
Inputs:
    filepath: filepath should include .pkl
    dict: dictionary to save
'''
def save_pickle_dict(filepath, dict):
    with open(filepath, 'wb') as f:
        pickle.dump(dict, f)

'''
Loads dictionary from pickle file at filepath and returns it
Inputs:
    filepath: filepath should end in .pkl
Returns:
    dictionary that was loaded
'''
def load_pickle_dict(filepath):
    with open(filepath, 'rb') as f:
        dict = pickle.load(f)
        return dict

'''
Saves dictionary as json file at filepath
Inputs:
    filepath: filepath should include .json
    dict: dictionary to save
'''
def save_json(filepath, dict):
    with open(filepath, "w") as f:
        json.dump(dict, f)

'''
Saves features as h5 file at filepath
Inputs:
    filepath: filepath should include .hdf5
    feature_dict: feature dict, (feature -> feature array)
'''
def save_h5(filepath, feature_dict):
    hf = h5py.File(filepath, 'w')
    for feature in feature_dict:
        hf.create_dataset(feature, data=feature_dict[feature])

'''
Loads txt file at filepath and returns it as numpy array
Inputs:
    filepath: filepath should end in .txt
Returns:
    numpy array containing each line as an item
'''
def load_txt(filepath):
    with open(filepath, 'r') as f:
        txt = f.read()
    return np.array([int(v) for v in txt.strip().split('\n')])

'''
Saves arr as txt file at filepath
Inputs:
    arr: array to save each item as new line
    filepath: filepath should include .txt
'''
def save_txt(filepath, arr):
    with open(filepath, 'w') as f:
        for item in arr:
            f.write(str(item) + "\n")

'''
Initializes variables needed for all preprocessing
Returns:
    config: base config
    datastack_name: base config datastack name
    mat_version: base config materialization version
    client: cave client
    data directory: data directory path
'''
def initialize():
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    datastack_name = config["data"]["datastack_name"]
    my_token = config["data"]["my_token"]
    mat_version = config["data"]["mat_version"]
    client = CAVEclient(datastack_name=datastack_name, auth_token=my_token, version=mat_version)
    data_directory = config["data"]["data_path"]
    return config, datastack_name, mat_version, client, data_directory

'''
Gets the correct list of roots for that chunk
Inputs:
    chunk_num: The chunk number or index + 1
    num_chunks: Number of total chunks to split the roots
Returns:
    root ids
'''
def get_roots_chunk(chunk_num, num_chunks):
    root_ids = get_roots()
    print("chunk number is:", chunk_num)
    print("num_chunks is:", num_chunks)
    chunk_size = len(root_ids) // num_chunks
    start_index = (chunk_num - 1) * chunk_size
    end_index = start_index + chunk_size + 1
    if chunk_num == num_chunks:
        root_ids = root_ids[start_index:]
    else:
        root_ids = root_ids[start_index:end_index]
    return root_ids


'''
Gets all root ids and returns them. 
Returns:
    root ids
'''
def get_roots():
    _, _, mat_version, _, data_directory = initialize()
    root_ids_path = f'{data_directory}/pre_edit_roots_list_{mat_version}.txt'
    root_ids = load_txt(root_ids_path)
    return root_ids