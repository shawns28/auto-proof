from caveclient import CAVEclient
import numpy as np
import pickle
import json
import argparse

CONFIG_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/base_config.json'
DATA_CONFIG_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/code/pre/data_config.json'

def get_config():
    """Initializes base config.

    Returns:
        config: base config
    """
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config

def get_data_config():
    """Initializes data config for pre-processing.
    
    Returns:
        data_config: base data config
    """
    with open(DATA_CONFIG_PATH, 'r') as f:
        data_config = json.load(f)
    return data_config

def create_client(data_config):
    """Initialize client and return it as well as relevant info.
    
    Args:
        data_config
    Returns:
        client: cave client
    """
    datastack_name = data_config["client"]["datastack_name"]
    my_token = data_config["client"]["my_token"]
    mat_version = data_config["client"]["mat_version_end"]
    client = CAVEclient(datastack_name=datastack_name, auth_token=my_token, version=mat_version)
    return client

def save_pickle_dict(filepath, dict):
    """Saves dictionary as pickle file at filepath
    
    Args:
        filepath: filepath should include .pkl
        dict: dictionary to save
    """
    with open(filepath, 'wb') as f:
        pickle.dump(dict, f)

def load_pickle_dict(filepath):
    """Loads dictionary from pickle file at filepath and returns it
    
    Args:
        filepath: filepath should end in .pkl
    Returns:
        dictionary that was loaded
    """
    with open(filepath, 'rb') as f:
        dict = pickle.load(f)
        return dict

def save_json(filepath, dict):
    """Saves dictionary as json file at filepath

    Args:
        filepath: filepath should include .json
        dict: dictionary to save
    """
    with open(filepath, "w") as f:
        json.dump(dict, f)

def load_txt(filepath):
    """Loads txt file at filepath and returns it as numpy array.

    Args:
        filepath: filepath should end in .txt
    Returns:
        numpy array containing each line as an item
    """
    with open(filepath, 'r') as f:
        txt = f.read()
    return np.array([v for v in txt.strip().split('\n')])

def save_txt(filepath, arr):
    """Saves arr as txt file at filepath.

    Args:
        arr: array to save each item as new line
        filepath: filepath should include .txt
    """
    with open(filepath, 'w') as f:
        for item in arr:
            f.write(str(item) + "\n")

def get_roots_chunk(root_ids, chunk_num, num_chunks):
    """Gets the correct list of roots for that chunk
    
    Args:
        config
        chunk_num: The chunk number or index + 1
        num_chunks: Number of total chunks to split the roots
    Returns:
        root ids
    """
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

def get_num_chunk_and_processes(data_config):
    """Gets command line flags for chunks and processes used for pre-processing.

    Args:
        data_config
    Returns:
        data_config
        chunk_num: Which chunk of roots
        num_chunks: Total number of chunks
        num_processes: Number of processes for multiprocessing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chunk_num", help="chunk num")
    parser.add_argument("-n", "--num_processes", help="num processes")
    args = parser.parse_args()
    if args.num_processes:
        data_config['multiprocessing']['num_processes'] = int(args.num_processes)
    num_processes = data_config['multiprocessing']['num_processes']
    if args.chunk_num:
        chunk_num = int(args.chunk_num)
        num_chunks = data_config['multiprocessing']['num_chunks']
    else: # No chunking
        num_chunks = 1
        chunk_num = 1
    return data_config, chunk_num, num_chunks, num_processes

def compare_roots(before_path, after_path, diff_path):
    """Compares the roots from before and after and saves the different roots"""
    # Comparing the roots after the operation to before
    before_op = load_txt(before_path)
    after_op = load_txt(after_path)
    diff = np.setdiff1d(before_op, after_op)
    print("different: ", len(diff))
    if len(diff) > 0:
        save_txt(diff_path, diff)

def add_identifier_to_roots(roots):
    """Adds _000 identifier to the end of each root"""
    new_roots = []
    for i in range(len(roots)):
        new_roots.append(str(roots[i]) + '_000')
    return new_roots