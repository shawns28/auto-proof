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
        # 'root' is all, 'train', 'val', 'test'
        self.roots = data_utils.load_txt(config['data'][f'{mode}_path'])
        # self.seed_index = config['loader']['seed_index']
        self.fov = config['loader']['fov']
        # self.num_shards = config['data']['num_shards']
        self.features_dir = config['data']['features_dir']
        self.map_pe_dir = config['data']['map_pe_dir']
        self.dist_dir = config['data']['dist_dir']
        self.rank_dir = config['data']['rank_dir']
        self.box_cutoff = config['data']['box_cutoff']
        # self.shard_features = config['data']['shard_features']
        # self.features_sharded_dir = config['data']['features_sharded_dir']

    def __len__(self):
        return len(self.roots)

    def get_root_index(self, root):
        index = np.where(self.roots == root)[0][0]
        return index
    
    def get_random_root(self):
        random_index = np.random.randint(0, len(self.roots))
        return self.roots[random_index]
    
    def get_is_proofread(self, root):
        data_path = f'{self.features_dir}{root}.hdf5'
        with h5py.File(data_path, 'r') as f:
            return f['is_proofread'][()]

    def get_num_initial_vertices(self, root):
        data_path = f'{self.features_dir}{root}.hdf5'
        with h5py.File(data_path, 'r') as f:
            return f['num_initial_vertices'][()]

    def __getitem__(self, index):
        root = self.roots[index]
        data_path = f'{self.features_dir}{root}.hdf5'
        map_pe_path = f'{self.map_pe_dir}map_{root}.hdf5'
        dist_path = f'{self.dist_dir}dist_{root}.hdf5'
        rank_path = f'{self.rank_dir}rank_{root}.hdf5'
        # NOTE: Sharded features don't have proofread roots in them
        # if self.shard_features: 
        #     shard_id = hash_shard(root, self.num_shards)
        #     data_path = f'{self.features_sharded_dir}{shard_id}.hdf5'
        try:
            with h5py.File(data_path, 'r') as f, h5py.File(map_pe_path, 'r') as map_pe_file, h5py.File(dist_path, 'r') as dist_file, h5py.File(rank_path, 'r') as rank_file:
                # f = shard_file
                # if self.shard_features:
                #     f = shard_file[str(root)]
                vertices = torch.from_numpy(f['vertices'][:])
                # Shouldn't be using compartment!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # compartment = torch.from_numpy(f['compartment'][:]).unsqueeze(1)
                radius = torch.from_numpy(f['radius'][:]).unsqueeze(1)
                labels = torch.from_numpy(f['label'][:]).unsqueeze(1).int()
                # print("original labels", labels)
                confidence = torch.from_numpy(f['confidence'][:]).unsqueeze(1).int()
                # print("num_initial_vertices", f['num_initial_vertices'][()])

                pos_enc = torch.from_numpy(map_pe_file['map_pe'][:])
                # pos_enc = torch.from_numpy(f['pos_enc'][:])

                # If doing the default features, need to set confidence to 1 where labels are 0
                # if not self.shard_features:
                #     confidence[labels == 0] = 1

                # Not adding rank as a feature
                # rank_num = f'rank_{self.seed_index}'
                # rank = torch.from_numpy(f[rank_num][:])
                rank = torch.from_numpy(rank_file[f'rank_{self.box_cutoff}'][:]).unsqueeze(1)

                dist_to_error = torch.from_numpy(dist_file['dist'][:]).unsqueeze(1)

                size = len(vertices)

                # Currently concatenating the pos enc to node features
                # Should remove the ones if not going to predict the padded ones
                # SHOULD REMOVE THE ONES NOW SINCE I USE -1 later to indicate that the label is padded anyway
                # input = torch.cat((vertices, compartment, radius, pos_enc, torch.ones(size, 1)), dim=1)
                # No longer use compartment or the ones
                input = torch.cat((vertices, radius, pos_enc), dim=1)

                edges = f['edges'][:]

                if size > self.fov:
                    indices = torch.where(rank < self.fov)[0]
                    input = input[indices]
                    labels = labels[indices]
                    confidence = confidence[indices]
                    dist_to_error = dist_to_error[indices]
                    rank = rank[indices]
                    edges = prune_edges(edges, indices)
                elif(size < self.fov):
                    input = torch.cat((input, torch.zeros((self.fov - size, input.shape[1]))), dim=0)
                    labels = torch.cat((labels, torch.full((self.fov - size, 1), -1)))
                    confidence = torch.cat((confidence, torch.full((self.fov - size, 1), -1)))
                    dist_to_error = torch.cat((dist_to_error, torch.full((self.fov - size, 1), -1)))
                    rank = torch.cat((rank, torch.full((self.fov - size, 1), -1)))

                adj = edge_list_to_adjency(edges, size, self.fov)
        except Exception as e:
            print("root: ", root, "error: ", e)
            return None
            
        return root, input, labels, confidence, dist_to_error, rank, adj

# def hash_shard(root, num_shards):
#     hash_value = hash(root)
#     hash_value = abs(hash_value)
#     shard_id = hash_value % num_shards
#     return shard_id

# Always adds in diagonals as self edges/loops
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