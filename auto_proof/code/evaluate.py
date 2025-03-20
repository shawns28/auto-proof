from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list_torch_skip_diag, prune_edges
from auto_proof.code.pre import data_utils
from auto_proof.code.model import create_model
from auto_proof.code.utils import get_root_output

import torch
import torch.nn as nn
import json
import numpy as np
from tqdm import tqdm
from torch.multiprocessing import Manager
import networkx as nx
import h5py
from torch.utils.data import Dataset, DataLoader, random_split

# NOT FINISHED

if __name__ == "__main__":
    ckpt_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/"   
    run_id = 'AUT-215'
    run_dir = f'{ckpt_dir}{run_id}/'
    ckpt_path = f'{run_dir}model_55.pt'
    with open(f'{run_dir}config.json', 'r') as f:
        config = json.load(f)

    data = AutoProofDataset(config, 'root')
    
    model = create_model(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    config['trainer']['thresholds'] = [0.01, 0.1, 0.4, 0.5, 0.6, 0.9, 0.99]
    config['trainer']['max_cloud'] = 15

    # Need to separate the methods in train into separate files because currently I can't even mimic it without copying the entire methods but we don't even need the methods to be self versioned
    