import torch
import pickle
import numpy as np
import glob
import h5py
import dataset
import json
from dataset import AutoProofDataset, build_dataloader
from model import create_model
import meshparty 

# root id 864691135463333789
root = '864691135463333789'
root_path = f'../../data/features/{root}_1000.h5py'
with h5py.File(root_path, 'r') as f:
    vertices = torch.from_numpy(f['vertices'][:])
    compartment = torch.from_numpy(f['compartment'][:]).unsqueeze(1)
    radius = torch.from_numpy(f['radius'][:]).unsqueeze(1)
    pos_enc = torch.from_numpy(f['pos_enc'][:])
    labels = torch.from_numpy(f['label'][:])
    confidence = torch.from_numpy(f['confidence'][:])
    # Need to add this to preprocessing instead of here to account for error locations as high conf
    confidence[labels == 0] = 1
