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

class AutoProofDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.roots = config['data']['root_path']
        self.seed_index = config['loader']['seed_index']
        self.fov = config['loader']['fov']
        self.num_shards = config['data']['num_shards']
        # print(self.roots[-50:])

    def __len__(self):
        return len(self.roots)

    def __getitem__(self, index):
        root = self.roots[index]
        shard_id = hash_shard(root, self.num_shards)
        shard_path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/sharded_features/{shard_id}.txt'
        try:
            with h5py.File(shard_path, 'r') as shard_file:
                f = shard_file[str(root)]
                vertices = torch.from_numpy(f['vertices'][:])
                compartment = torch.from_numpy(f['compartment'][:]).unsqueeze(1)
                radius = torch.from_numpy(f['radius'][:]).unsqueeze(1)
                pos_enc = torch.from_numpy(f['pos_enc'][:])
                labels = torch.from_numpy(f['label'][:])
                confidence = torch.from_numpy(f['confidence'][:])

                labels = labels.unsqueeze(1).int()
                confidence = confidence.unsqueeze(1).int()

                # Not adding rank as a feature
                rank_num = f'rank_{self.seed_index}'
                rank = torch.from_numpy(f[rank_num][:])

                size = len(vertices)

                # Currently concatenating the pos enc to node features
                # Should remove the ones if not going to predict the padded ones
                # SHOULD REMOVE THE ONES NOW SINCE I USE -1 later to indicate that the label is padded anyway
                input = torch.cat((vertices, compartment, radius, pos_enc, torch.ones(size, 1)), dim=1)

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

def hash_shard(root, num_shards):
    hash_value = hash(root)
    hash_value = abs(hash_value)
    shard_id = hash_value % num_shards
    return shard_id

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
    adj = np.zeros((size, size))
    for edge in edges:
        adj[edge[0]][edge[1]] = 1
        adj[edge[1]][edge[0]] = 1
        adj[edge[0]][edge[0]] = 1
        adj[edge[1]][edge[1]] = 1
    return adj

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


def build_dataloader(dataset, config):
    num_workers = max(config['loader']['num_workers'], os.cpu_count() // 2)
    print("num workers", num_workers)

    batch_size = config['loader']['batch_size']
    print("batch size" , batch_size)

    # prefetch_factor = config['loader']['prefetch_factor']
    # prefetch_factor = 1
    # loader = DataLoader(
    #         ProofDataset(config),
    #         batch_size=batch_size, 
    #         shuffle=False,
    #         num_workers=num_workers,
    #         pin_memory=True,
    #         persistent_workers=True,
    #         prefetch_factor=prefetch_factor)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) 

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True)

    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True)

    return train_loader, val_loader, train_size, val_size