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
        self.roots = data_utils.load_txt(config['data'][f'{mode}_path'])[:6000]
        self.seed_index = config['loader']['seed_index']
        self.fov = config['loader']['fov']
        self.num_shards = config['data']['num_shards']
        self.features_dir = config['data']['features_dir']
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
        # NOTE: Sharded features don't have proofread roots in them
        # if self.shard_features: 
        #     shard_id = hash_shard(root, self.num_shards)
        #     data_path = f'{self.features_sharded_dir}{shard_id}.hdf5'
        try:
            with h5py.File(data_path, 'r') as shard_file:
                f = shard_file
                # if self.shard_features:
                #     f = shard_file[str(root)]
                vertices = torch.from_numpy(f['vertices'][:])
                # Shouldn't be using compartment!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # compartment = torch.from_numpy(f['compartment'][:]).unsqueeze(1)
                radius = torch.from_numpy(f['radius'][:]).unsqueeze(1)
                pos_enc = torch.from_numpy(f['pos_enc'][:])
                labels = torch.from_numpy(f['label'][:])
                # print("original labels", labels)
                confidence = torch.from_numpy(f['confidence'][:])
                # print("num_initial_vertices", f['num_initial_vertices'][()])
                labels = labels.unsqueeze(1).int()
                confidence = confidence.unsqueeze(1).int()

                # If doing the default features, need to set confidence to 1 where labels are 0
                # if not self.shard_features:
                #     confidence[labels == 0] = 1

                # Not adding rank as a feature
                rank_num = f'rank_{self.seed_index}'
                rank = torch.from_numpy(f[rank_num][:])

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
                    edges = prune_edges(edges, indices)
                elif(size < self.fov):
                    input = torch.cat((input, torch.zeros((self.fov - size, input.shape[1]))), dim=0)
                    labels = torch.cat((labels, torch.full((self.fov - size, 1), -1)))
                    confidence = torch.cat((confidence, torch.full((self.fov - size, 1), -1)))

                adj = edge_list_to_adjency(edges, self.fov)
        except Exception as e:
            print("root: ", root, "error: ", e)
            return None
            
        return input, labels, confidence, adj

# def hash_shard(root, num_shards):
#     hash_value = hash(root)
#     hash_value = abs(hash_value)
#     shard_id = hash_value % num_shards
#     return shard_id

# 0.002 sec but double count diagonal
def edge_list_to_adjency_numpy(edges, size):
    adj = np.zeros((size, size))
    edges_x = edges[:, 0]
    edges_y = edges[:, 1]
    np.add.at(adj, (edges_x, edges_y), 1)
    np.add.at(adj, (edges_y, edges_x), 1)
    np.add.at(adj, (edges_x, edges_x), 1)
    np.add.at(adj, (edges_y, edges_y), 1)
    return adj

# 0.003 sec
def edge_list_to_adjency(edges, size):
    adj = torch.zeros((size, size))
    for edge in edges:
        adj[edge[0]][edge[1]] = 1
        adj[edge[1]][edge[0]] = 1
        adj[edge[0]][edge[0]] = 1
        adj[edge[1]][edge[1]] = 1
    return adj

def adjency_to_edge_list(adj):
    edge_list = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if j >= i and adj[i][j] != 0:
                edge_list.append((i, j))
    return edge_list

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

def build_dataloader(config, dataset, run, mode):
    num_workers = config['loader']['num_workers']
    batch_size = config['loader']['batch_size']
    shuffle = False

    if mode == 'root' or mode == 'train':
        run["parameters/loader/num_workers"] = num_workers
        print("num workers", num_workers)
        print("batch size" , batch_size)
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