from caveclient import CAVEclient
import numpy as np
import pickle
import json

CONFIG_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/base_config.json'

'''
Initializes config
Returns:
    config: base config
'''
def get_config():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config

'''
Initialize client and return it as well as relevant info
Inputs:
    config
Returns:
    client: cave client
    datastack_name: base config datastack name
    mat_version: base config materialization version
'''
def create_client(config):
    datastack_name = config["data"]["datastack_name"]
    my_token = config["data"]["my_token"]
    mat_version = config["data"]["mat_version"]
    client = CAVEclient(datastack_name=datastack_name, auth_token=my_token, version=mat_version)
    return client, datastack_name, mat_version

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
Loads txt file at filepath and returns it as numpy array
Inputs:
    filepath: filepath should end in .txt
Returns:
    numpy array containing each line as an item
'''
def load_txt(filepath):
    with open(filepath, 'r') as f:
        txt = f.read()
    return np.array([v for v in txt.strip().split('\n')])

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
Gets the correct list of roots for that chunk
Inputs:
    config
    chunk_num: The chunk number or index + 1
    num_chunks: Number of total chunks to split the roots
Returns:
    root ids
'''
def get_roots_chunk(config, root_ids, chunk_num, num_chunks):
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
Compares the roots from before and after and saves the different roots
Inputs:
    before_path
    after_path
    diff_path
'''
def compare_roots(before_path, after_path, diff_path):
    # Comparing the roots after the operation to before
    before_op = load_txt(before_path)
    after_op = load_txt(after_path)
    diff = np.setdiff1d(before_op, after_op)
    print("different: ", len(diff))
    if len(diff) > 0:
        save_txt(diff_path, diff)