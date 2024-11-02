import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import h5py
import os
from torch.multiprocessing import Manager
from cloudvolume import CloudVolume

class AutoProofDataset(Dataset):
    def __init__(self, config):
        self.config = config
        files = glob.glob('../../data/successful_labels/*')
        files = [files[0]] # 864691135778235581
        self.roots = [files[i][-18:] for i in range(len(files))]
        self.seed_index = config['loader']['seed_index']

    def __len__(self):
        return len(self.roots)

    def __getitem__(self, index):
        root = self.roots[index]
        root_path = f'../../data/features/{root}_1000.h5py'
        with h5py.File(root_path, 'r') as f:
            vertices = torch.from_numpy(f['vertices'][:])
            compartment = torch.from_numpy(f['compartment'][:]).unsqueeze(1)
            radius = torch.from_numpy(f['radius'][:]).unsqueeze(1)
            labels = torch.from_numpy(f['label'][:]).unsqueeze(1)
            confidence = torch.from_numpy(f['confidence'][:]).unsqueeze(1)
            pos_enc = torch.from_numpy(f['pos_enc'][:])

            # Not adding rank as a feature
            rank_num = f'rank_{self.seed_index}'
            rank = f[rank_num][:]

            # Edges information is used in adjency matrix and pos_enc
            edges = torch.from_numpy(f['edges'][:])

            input = torch.cat((vertices, compartment, radius, labels, confidence, pos_enc), dim=1)

            print(vertices.shape, edges.shape, compartment.shape, radius.shape, labels.shape, confidence.shape, pos_enc.shape, rank.shape)
            print("input shape", input.shape)
        return [1]

class ProofDataset(Dataset):
    def __init__(self, config, inference=False):

        self.config = config
        self.inference = inference
        data_directory = '../../data'
        features_directory = f'{data_directory}/features'
        self.root_paths = glob.glob(f'{features_directory}/*')
        # very bad because this is based off of the path which could always change
        self.roots = [self.root_paths[i][20:38] for i in range(len(self.root_paths))]
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
        converted_features_path = '../../data/features_converted_1000'

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

class VerticesConvertedDataset(Dataset):
    def __init__(self, config, inference=False):

        self.config = config
        converted_features_path = '../../data/features_converted_1000'

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

class VerticesDataset(Dataset):
    def __init__(self, config, inference=False):
        self.config = config
        features_path = '../../data/features'
        self.root_paths = glob.glob(f'{features_path}/*')
        self.num_roots = len(self.root_paths)
        # print(self.num_roots)

    def __len__(self):
        return self.num_roots
    
    def __getitem__(self, index): 
        with h5py.File(self.root_paths[index], 'r') as f:
            vertices = f['vertices'][:],
        root = self.root_paths[index][20:38]
        # print("root", root)
        return root, vertices

class VerticesParallelDataset(Dataset):
    def __init__(self, config, inference=False):
        self.config = config
        features_path = '../../data/features'
        self.root_paths = glob.glob(f'{features_path}/*')
        self.num_roots = len(self.root_paths)
        # print(self.num_roots)
        seg_path = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1"
        self.cv_seg = CloudVolume(seg_path, use_https=True, fill_missing=True)
        self.resolution = np.array([[8, 8, 40]])

    def __len__(self):
        return self.num_roots
    
    def __getitem__(self, index):
        print(self.root_paths[index])
        with h5py.File(self.root_paths[index], 'r') as f:
            vertices = np.array(f['vertices'][:]),
            vertices = vertices[0]
            # print(vertices)
            print(len(vertices))
            vertices = vertices / self.resolution
            # print("vertices", vertices)
            svs = np.empty(len(vertices), dtype=int)
            for i, vertice in enumerate(vertices):
                vol = self.cv_seg[vertice[0], vertice[1], vertice[2]]
                svs[i] = vol[0][0][0][0]
            print("svs", svs)
            return svs

class ProofConvertedDataset(Dataset):
    def __init__(self, config, inference=False, vertices=False, ):

        self.config = config
        self.inference = inference
        converted_features_path = '../../data/features_converted_gzip_1000'

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

def build_dataloader(dataset):
    # num_workers = config['loader']['num_workers']
    num_workers = 2

    # batch_size = config['loader']['batch_size']
    batch_size = 1

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
    
    loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True)

    return loader