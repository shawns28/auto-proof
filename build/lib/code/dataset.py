import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import h5py
import os
from torch.multiprocessing import Manager

class ProofDataset(Dataset):
    def __init__(self, config, inference=False):

        self.config = config
        self.inference = inference
        data_directory = '../data'
        features_directory = f'{data_directory}/features'
        self.root_paths = glob.glob(f'{features_directory}/*')
        # very bad because this is based off of the path which could always change
        self.roots = [self.root_paths[i][17:35] for i in range(len(self.root_paths))]
        self.num_roots = len(self.roots)

    def __len__(self):
        return self.num_roots
    
    
    def __getitem__(self, index): 
        root = self.roots[index]
        root_path = self.root_paths[index]
        # print("root path", root_path)
        with h5py.File(root_path, 'r') as f:
            vertices = f['vertices'][:],
            edges = f['edges'][:],
            rank_0 = f['rank_0'][:],
            rank_1 = f['rank_1'][:],
            rank_2 = f['rank_2'][:],
            num_vertices = f['num_vertices'][()],
            num_initial_vertices = f['num_initial_vertices'][()],
            compartment = f['compartment'][:],
            radius = f['radius'][:]

        return root, vertices, edges, rank_0, rank_1, rank_2, num_vertices, num_initial_vertices, compartment, radius

class ProofInMemoryDataset(Dataset):
    def __init__(self, config, inference=False):

        self.config = config
        self.inference = inference
        converted_features_path = '../data/features_converted_1000'

        self.hf_vertices = h5py.File(f'{converted_features_path}/vertices.hdf5', 'r')
        self.hf_edges = h5py.File(f'{converted_features_path}/edges.hdf5', 'r')
        self.hf_rank_0 = h5py.File(f'{converted_features_path}/rank_0.hdf5', 'r')
        self.hf_rank_1 = h5py.File(f'{converted_features_path}/rank_1.hdf5', 'r')
        self.hf_rank_2 = h5py.File(f'{converted_features_path}/rank_2.hdf5', 'r')
        self.hf_num_vertices = h5py.File(f'{converted_features_path}/num_vertices.hdf5', 'r')
        self.hf_num_initial_vertices = h5py.File(f'{converted_features_path}/num_initial_vertices.hdf5', 'r')
        self.hf_compartment = h5py.File(f'{converted_features_path}/compartment.hdf5', 'r')
        self.hf_radius = h5py.File(f'{converted_features_path}/radius.hdf5', 'r')

        self.roots = list(self.hf_num_vertices.keys())
        # print(self.roots)
        self.num_roots = len(self.roots)
        # print(self.num_roots)

        self.manager = Manager()
        self.dict = self.manager.dict()
        with tqdm(total=self.num_roots) as pbar:
            for i, root in enumerate(self.roots):
                features = {
                    'root': root,
                    'vertices': self.hf_vertices[root][:],
                    'edges': self.hf_edges[root][:],
                    'rank_0': self.hf_rank_0[root][:],
                    'rank_1': self.hf_rank_1[root][:],
                    'rank_2': self.hf_rank_2[root][:],
                    'num_vertices': self.hf_num_vertices[root][()],
                    'num_initial_vertices': self.hf_num_initial_vertices[root][()],
                    'compartment': self.hf_compartment[root][:],
                    'radius': self.hf_radius[root][:]
                }
                self.dict[i] = features
                pbar.update()
    def __len__(self):
        return self.num_roots
    
    
    def __getitem__(self, index): 
        features = self.dict[index]
        root = features['root']
        vertices = features['vertices']
        edges = features['edges']
        rank_0 = features['rank_0']
        rank_1 = features['rank_1']
        rank_2 = features['rank_2']
        num_vertices = features['num_vertices']
        num_initial_vertices = features['num_intitial_vertices']
        compartment = features['compartments']
        radius = features['radius']
        return root, vertices, edges, rank_0, rank_1, rank_2, num_vertices, num_initial_vertices, compartment, radius

class VerticesDataset(Dataset):
    def __init__(self, config, inference=False):

        self.config = config
        converted_features_path = '../data/features_converted_1000'

        self.hf_vertices = h5py.File(f'{converted_features_path}/vertices.hdf5', 'r')

        self.roots = list(self.hf_vertices.keys())
        # print(self.roots)
        self.num_roots = len(self.roots)
        # print(self.num_roots)

    def __len__(self):
        return self.num_roots
    
    def __getitem__(self, index): 
        root = self.roots[index]
        # print("root", root)
        vertices = self.hf_vertices[root][:]
        
        return root, vertices


class ProofConvertedDataset(Dataset):
    def __init__(self, config, inference=False, vertices=False, ):

        self.config = config
        self.inference = inference
        converted_features_path = '../data/features_converted_1000'

        self.hf_vertices = h5py.File(f'{converted_features_path}/vertices.hdf5', 'r')
        self.hf_edges = h5py.File(f'{converted_features_path}/edges.hdf5', 'r')
        self.hf_rank_0 = h5py.File(f'{converted_features_path}/rank_0.hdf5', 'r')
        self.hf_rank_1 = h5py.File(f'{converted_features_path}/rank_1.hdf5', 'r')
        self.hf_rank_2 = h5py.File(f'{converted_features_path}/rank_2.hdf5', 'r')
        self.hf_num_vertices = h5py.File(f'{converted_features_path}/num_vertices.hdf5', 'r')
        self.hf_num_initial_vertices = h5py.File(f'{converted_features_path}/num_initial_vertices.hdf5', 'r')
        self.hf_compartment = h5py.File(f'{converted_features_path}/compartment.hdf5', 'r')
        self.hf_radius = h5py.File(f'{converted_features_path}/radius.hdf5', 'r')

        self.roots = list(self.hf_num_vertices.keys())
        # print(self.roots)
        self.num_roots = len(self.roots)
        # print(self.num_roots)

    def __len__(self):
        return self.num_roots
    
    
    def __getitem__(self, index): 
        root = self.roots[index]
        # print("root", root)
        vertices = self.hf_vertices[root][:]
        edges = self.hf_edges[root][:]
        rank_0 = self.hf_rank_0[root][:]
        rank_1 = self.hf_rank_1[root][:]
        rank_2 = self.hf_rank_2[root][:]
        num_vertices = self.hf_num_vertices[root][()]
        num_initial_vertices = self.hf_num_initial_vertices[root][()]
        compartment = self.hf_compartment[root][:]
        radius = self.hf_radius[root][:]
        
        return root, vertices, edges, rank_0, rank_1, rank_2, num_vertices, num_initial_vertices, compartment, radius

def build_dataloader(config):
    # num_workers = config['data']['num_workers']
    num_workers = 32

    # batch_size = config['data']['batch_size']
    batch_size = 1

    # prefetch_factor = config['data']['prefetch_factor']
    # prefetch_factor = 1
    # loader = DataLoader(
    #         ProofDataset(config),
    #         batch_size=batch_size, 
    #         shuffle=False,
    #         num_workers=num_workers,
    #         pin_memory=True,
    #         persistent_workers=True,
    #         prefetch_factor=prefetch_factor)
    
    loader = DataLoader(
            ProofDataset(config),
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True)

    return loader