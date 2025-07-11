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
    """
    A PyTorch Dataset class for loading root information.

    This dataset handles loading features, labels, and SegCLR
    features from HDF5 files, applying various normalizations,
    and preparing the data for training.
    """
    def __init__(self, config, mode):
        """
        Initializes the AutoProofDataset.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): The mode of the dataset, e.g., 'train', 'val', 'test', or 'all'. 
                        'all' is used for all roots. 
        """
        self.config = config
        self.data_dir = config['data']['data_dir']
        mat_version_start = config['data']['mat_version_start']
        mat_version_end = config['data']['mat_version_end']
        self.roots_dir = f'{self.data_dir}roots_{mat_version_start}_{mat_version_end}/'
        self.split_dir = f'{self.roots_dir}{config['data']['split_dir']}'
        self.roots = data_utils.load_txt(f'{self.split_dir}{config['data'][f'{mode}_roots']}')
        self.labels_dir = f'{self.data_dir}{config['data']['labels_dir']}'
        self.features_dir = f'{self.data_dir}{config['data']['features_dir']}'
        self.segclr_dir = f'{self.data_dir}{config['data']['segclr_dir']}'
        self.proofread_roots = data_utils.load_txt(f'{self.data_dir}{config['data']['proofread_dir']}{config['data']['proofread_roots']}')

        self.fov = config['loader']['fov']
        self.box_cutoff = config['data']['box_cutoff']
        self.use_segclr = config['loader']['use_segclr']
        self.relative_vertices = config['loader']['relative_vertices']
        self.zscore_radius = config['loader']['zscore_radius']
        self.zscore_segclr = config['loader']['zscore_segclr']
        self.l2_norm_segclr = config['loader']['l2_norm_segclr']
        self.zscore_pe = config['loader']['zscore_pe']

        self.radius_mean = torch.from_numpy(np.load(f'{self.split_dir}radius_mean.npy'))
        self.radius_std = torch.from_numpy(np.load(f'{self.split_dir}radius_std.npy'))
        self.pos_mean = torch.from_numpy(np.load(f'{self.split_dir}pos_mean.npy'))
        self.pos_std = torch.from_numpy(np.load(f'{self.split_dir}pos_std.npy'))
        self.segclr_mean = torch.from_numpy(np.load(f'{self.split_dir}segclr_mean.npy'))
        self.segclr_std = torch.from_numpy(np.load(f'{self.split_dir}segclr_std.npy'))

    def __len__(self):
        """
        Returns the total number of roots in the dataset.

        Returns:
            int: The number of roots.
        """
        return len(self.roots)

    def get_root_index(self, root):
        """
        Returns the index of a given root in the dataset.

        Args:
            root (str): The root identifier.

        Returns:
            int: The index of the root.
        """
        index = np.where(self.roots == root)[0][0]
        return index
    
    def get_random_root(self):
        """
        Returns a random root identifier from the dataset.

        Returns:
            str: A random root identifier.
        """
        random_index = np.random.randint(0, len(self.roots))
        return self.roots[random_index]
    
    def get_is_proofread(self, root):
        """
        Checks if a given root has been proofread.

        Args:
            root (str): The root identifier.

        Returns:
            bool: True if the root has been proofread, False otherwise.
        """
        root_without_ident = root[:-4]
        return root_without_ident in self.proofread_roots        

    def get_num_initial_vertices(self, root):
        """
        Retrieves the number of initial vertices for a given root.

        Args:
            root (str): The root identifier.

        Returns:
            int: The number of initial vertices.
        """
        data_path = f'{self.features_dir}{root}.hdf5'
        with h5py.File(data_path, 'r') as f:
            return f['num_initial_vertices'][()]

    def __getitem__(self, index):
        """
        Retrieves a single item from the dataset at the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - root (str): The root identifier.
                - input (torch.Tensor): The concatenated input features.
                - labels (torch.Tensor): The ground truth labels.
                - confidences (torch.Tensor): Confidence scores for labels.
                - dist_to_error (torch.Tensor): Distance to the nearest error.
                - rank (torch.Tensor): Rank of each vertex.
                - adj (torch.Tensor): Adjacency matrix of the graph.
                - mean_vertices (torch.Tensor): Mean of vertices (used for visualization).
            Returns None if there's an error loading the data for a root.
        """
        root = self.roots[index]
        feature_path = f'{self.features_dir}{root}.hdf5'
        labels_path = f'{self.labels_dir}{root}.hdf5'
        segclr_path = f'{self.segclr_dir}{root}.hdf5'
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

                mean_vertices = torch.zeros(3)
                if self.relative_vertices:
                    mean_vertices = torch.mean(vertices, dim=0, keepdim=True) # normalize vertices
                    vertices = vertices - mean_vertices

                if self.zscore_radius:
                    radius = (radius - self.radius_mean) / self.radius_std

                if self.zscore_pe:
                    pos_enc = (pos_enc - self.pos_mean) / self.pos_std

                if self.use_segclr:
                    with h5py.File(segclr_path, 'r') as segclr_f:
                        segclr = torch.from_numpy(segclr_f['segclr'][:]).float()
                        if self.l2_norm_segclr:
                            segclr = torch.nn.functional.normalize(segclr, p=2, dim=1)

                        if self.zscore_segclr:
                            segclr = (segclr - self.segclr_mean) / self.segclr_std
                        has_emb = torch.from_numpy(segclr_f['has_emb'][:]).unsqueeze(1)
                        input = torch.cat((vertices, radius, pos_enc, segclr, has_emb), dim=1)
                else:
                    input = torch.cat((vertices, radius, pos_enc), dim=1)
                
                size = len(vertices)
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
            
        return root, input, labels, confidences, dist_to_error, rank, adj, mean_vertices

def edge_list_to_adjency(edges, size, fov):
    """
    Converts a list of edges to an adjacency matrix.

    Always adds in diagonals as self edges/loops.

    Args:
        edges (np.ndarray): A 2D numpy array where each row represents an edge (u, v).
        size (int): The original number of vertices in the graph.
        fov (int): The field of view, which determines the size of the adjacency matrix.

    Returns:
        torch.Tensor: An (fov x fov) adjacency matrix as a PyTorch tensor.
    """
    adj = torch.zeros((fov, fov))
    adj[edges[:, 0], edges[:, 1]] = 1
    adj[edges[:, 1], edges[:, 0]] = 1
    for i in range(min(size, fov)):
        adj[i, i] = 1
    return adj

def adjency_to_edge_list_torch_skip_diag(adj):
    """
    Converts an adjacency matrix to an edge list, skipping diagonal (self-loop) edges.

    Args:
        adj (torch.Tensor): An adjacency matrix.

    Returns:
        torch.Tensor: A 2D PyTorch tensor where each row is an edge (u, v),
                      with u < v to avoid duplicates for undirected graphs.
    """
    rows, cols = torch.where(adj != 0)
    edges = torch.stack((rows, cols), dim=1)

    # Removes diagonals and duplicates since adj is undirected
    mask = edges[:, 0] < edges[:, 1]
    edges_upper = edges[mask]
    return edges_upper

def adjency_to_edge_list_numpy_skip_diag(adj):
    """
    Converts an adjacency matrix to an edge list using NumPy, skipping diagonal (self-loop) edges.

    Args:
        adj (np.ndarray): An adjacency matrix.

    Returns:
        np.ndarray: A 2D NumPy array where each row is an edge (u, v),
                    with u < v to avoid duplicates for undirected graphs.
    """
    rows, cols = np.where(adj != 0)
    edges = np.column_stack((rows, cols))

    # Removes diagonals and duplicates since adj is undirected
    mask = edges[:, 0] < edges[:, 1]
    edges_upper = edges[mask]
    return edges_upper

def prune_edges(edges, indices):
    """
    Prunes an edge list to include only edges whose endpoints are within the given indices.
    Also remaps the original vertex indices to new, contiguous indices based on the `indices` list.

    Args:
        edges (np.ndarray): A 2D numpy array where each row represents an edge (u, v).
        indices (torch.Tensor): A 1D tensor of original vertex indices that are to be kept.

    Returns:
        np.ndarray: A new 2D numpy array of pruned and remapped edges.
    """
    orig_to_new = {value.item(): index for index, value in enumerate(indices)}
    new_edges = []
    for i in range(len(edges)):
        edge = edges[i]
        if edge[0] in orig_to_new and edge[1] in orig_to_new:
            new_edges.append([orig_to_new[edge[0]], orig_to_new[edge[1]]])
    return np.array(new_edges)

def build_dataloader(config, dataset, mode):
    """
    Builds a PyTorch DataLoader for the given dataset.

    Args:
        config (dict): A dictionary containing configuration parameters for the DataLoader.
        dataset (torch.utils.data.Dataset): The dataset to load.
        mode (str): The mode of the dataset, e.g., 'train', 'val', 'test', or 'all'.
                    'train' and 'all' modes will shuffle the data.

    Returns:
        torch.utils.data.DataLoader: A configured PyTorch DataLoader.
    """
    num_workers = config['loader']['num_workers']
    batch_size = config['loader']['batch_size']
    shuffle = False

    if mode == 'all' or mode == 'train':
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