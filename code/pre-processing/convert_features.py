import torch
import pickle
import numpy as np
from tqdm import tqdm
import glob
import h5py
import dataset
import json

def convert_features():
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    data = dataset.ProofDataset(config)
    loader = dataset.build_dataloader(data)
    print("size of data", data.__len__())
    converted_features_path = '../../data/features_converted_1000'
    hf_vertices = h5py.File(f'{converted_features_path}/vertices.hdf5', 'w-')
    hf_edges = h5py.File(f'{converted_features_path}/edges.hdf5', 'w-')
    hf_rank_0 = h5py.File(f'{converted_features_path}/rank_0.hdf5', 'w-')
    hf_rank_1 = h5py.File(f'{converted_features_path}/rank_1.hdf5', 'w-')
    hf_rank_2 = h5py.File(f'{converted_features_path}/rank_2.hdf5', 'w-')
    hf_num_vertices = h5py.File(f'{converted_features_path}/num_vertices.hdf5', 'w-')
    hf_num_initial_vertices = h5py.File(f'{converted_features_path}/num_initial_vertices.hdf5', 'w-')
    hf_compartment = h5py.File(f'{converted_features_path}/compartment.hdf5', 'w-')
    hf_radius = h5py.File(f'{converted_features_path}/radius.hdf5', 'w-')

    with tqdm(total=data.__len__()) as pbar:
        for data in loader:
            root, vertices, edges, rank_0, rank_1, rank_2, num_vertices, num_initial_vertices, compartment, radius = data
            root = root[0]
            vertices = vertices[0][0]
            edges = edges[0][0]
            rank_0 = rank_0[0][0]
            rank_1 = rank_1[0][0]
            rank_2 = rank_2[0][0]
            num_vertices = num_vertices[0][0]
            num_initial_vertices = num_initial_vertices[0][0]
            compartment = compartment[0][0]
            radius = radius[0][0]
            hf_vertices.create_dataset(root, data=vertices)
            hf_edges.create_dataset(root, data=edges)
            hf_rank_0.create_dataset(root, data=rank_0)
            hf_rank_1.create_dataset(root, data=rank_1)
            hf_rank_2.create_dataset(root, data=rank_2)
            hf_num_vertices.create_dataset(root, data=num_vertices)
            hf_num_initial_vertices.create_dataset(root, data=num_initial_vertices)
            hf_compartment.create_dataset(root, data=compartment)
            hf_radius.create_dataset(root, data=radius)
            pbar.update()

    hf_edges.close()
    hf_rank_0.close()
    hf_rank_1.close()
    hf_rank_2.close()
    hf_num_vertices.close()
    hf_num_initial_vertices.close()
    hf_compartment.close()
    hf_radius.close()

if __name__ == "__main__":
    convert_features()