from auto_proof.code.pre import data_utils

import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import h5py
import os
from torch.multiprocessing import Manager
import time
import neptune

class AutoProofDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.data_dir = config['data']['data_dir']
        mat_version_start = config['data']['mat_version_start']
        mat_version_end = config['data']['mat_version_end']
        self.roots_dir = f'{self.data_dir}roots_{mat_version_start}_{mat_version_end}/'
        self.split_dir = f'{self.roots_dir}{config['data']['split_dir']}'
        self.roots = data_utils.load_txt(f'{self.split_dir}{config['data'][f'{mode}_roots']}')
        self.labels_dir = f'{self.data_dir}{config['data']['labels_dir']}'
        self.features_dir = f'{self.data_dir}{config['data']['features_dir']}'
        self.proofread_roots = data_utils.load_txt(f'{self.data_dir}{config['data']['proofread_dir']}{config['data']['proofread_roots']}')
       
        self.fov = config['loader']['fov']
        self.box_cutoff = config['data']['box_cutoff']

    def __len__(self):
        return len(self.roots)

    def get_root_index(self, root):
        index = np.where(self.roots == root)[0][0]
        return index
    
    def get_random_root(self):
        random_index = np.random.randint(0, len(self.roots))
        return self.roots[random_index]
    
    # TODO: Change how this works because it won't work anymore
    def get_is_proofread(self, root):
        root_without_ident = root[:-4]
        return root_without_ident in self.proofread_roots        

    def get_num_initial_vertices(self, root):
        data_path = f'{self.features_dir}{root}.hdf5'
        with h5py.File(data_path, 'r') as f:
            return f['num_initial_vertices'][()]

    def __getitem__(self, index):
        root = self.roots[index]
        feature_path = f'{self.features_dir}{root}.hdf5'
        labels_path = f'{self.labels_dir}{root}.hdf5'
        try:
            with h5py.File(feature_path, 'r') as feat_f,  h5py.File(labels_path, 'r') as labels_f:
                vertices = torch.from_numpy(feat_f['vertices'][:])
                pos_enc = torch.from_numpy(feat_f['map_pe'][:])

                rank = torch.from_numpy(feat_f[f'rank'][:]).unsqueeze(1)
                radius = torch.from_numpy(feat_f['radius'][:]).unsqueeze(1)
                edges = feat_f['edges'][:]

                dist_to_error = torch.from_numpy(labels_f['dist'][:]).unsqueeze(1)
                labels = torch.from_numpy(labels_f['labels'][:]).int().unsqueeze(1)
                confidences = torch.from_numpy(labels_f['confidences'][:]).int().unsqueeze(1)

                size = len(vertices)

                input = torch.cat((vertices, radius, pos_enc), dim=1)

                if size > self.fov:
                    indices = torch.where(rank < self.fov)[0]
                    input = input[indices]
                    labels = labels[indices]
                    confidences = confidences[indices]
                    dist_to_error = dist_to_error[indices]
                    rank = rank[indices]
                    edges = prune_edges(edges, indices)
                elif(size < self.fov):
                    input = torch.cat((input, torch.zeros((self.fov - size, input.shape[1]))), dim=0)
                    labels = torch.cat((labels, torch.full((self.fov - size, 1), -1)))
                    confidences = torch.cat((confidences, torch.full((self.fov - size, 1), -1)))
                    dist_to_error = torch.cat((dist_to_error, torch.full((self.fov - size, 1), -1)))
                    rank = torch.cat((rank, torch.full((self.fov - size, 1), -1)))

                adj = edge_list_to_adjency(edges, size, self.fov)
        except Exception as e:
            print("root: ", root, "error: ", e)
            return None
            
        return root, input, labels, confidences, dist_to_error, rank, adj

# def hash_shard(root, num_shards):
#     hash_value = hash(root)
#     hash_value = abs(hash_value)
#     shard_id = hash_value % num_shards
#     return shard_id

# Always adds in diagonals as self edges/loops
# Returns a torch tensor
def edge_list_to_adjency(edges, size, fov):
    adj = torch.zeros((fov, fov))
    adj[edges[:, 0], edges[:, 1]] = 1
    adj[edges[:, 1], edges[:, 0]] = 1
    for i in range(min(size, fov)):
        adj[i, i] = 1
    return adj

# Skips edge pairs for diagonals
def adjency_to_edge_list_torch_skip_diag(adj):
    rows, cols = torch.where(adj != 0)
    edges = torch.stack((rows, cols), dim=1)

    # Removes diagonals and duplicates since adj is undirected
    mask = edges[:, 0] < edges[:, 1]
    edges_upper = edges[mask]
    return edges_upper

# Skips edge pairs for diagonals
def adjency_to_edge_list_numpy_skip_diag(adj):
    rows, cols = np.where(adj != 0)
    edges = np.column_stack((rows, cols))

    # Removes diagonals and duplicates since adj is undirected
    mask = edges[:, 0] < edges[:, 1]
    edges_upper = edges[mask]
    return edges_upper

# adapted from skeletonize.py
def prune_edges(edges, indices):
    orig_to_new = {value.item(): index for index, value in enumerate(indices)}
    edge_mask = np.zeros(len(edges), dtype=bool)
    for i in range(len(edges)):
        edge = edges[i]
        if edge[0] in orig_to_new and edge[1] in orig_to_new:
            edges[i][0] = orig_to_new[edge[0]]
            edges[i][1] = orig_to_new[edge[1]]
            edge_mask[i] = True
    return edges[edge_mask]

def build_dataloader(config, dataset, mode):
    num_workers = config['loader']['num_workers']
    batch_size = config['loader']['batch_size']
    shuffle = False

    if mode == 'root' or mode == 'train':
        shuffle = True

    prefetch_factor = config['loader']['prefetch_factor']
    if prefetch_factor == 0:
        prefetch_factor = None
    
    return DataLoader(
            dataset=dataset,
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor)