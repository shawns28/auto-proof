import torch
import pickle
import numpy as np
from tqdm import tqdm
import glob
import h5py
import dataset
import json

def main():
    print("main original")
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    data = dataset.ProofDataset(config)
    loader = dataset.build_dataloader(data)
    print("size of data", data.__len__())
    with tqdm(total=data.__len__()) as pbar:
        count = 0
        for item in loader:
            root, vertices, edges, rank_0, rank_1, rank_2, num_vertices, num_initial_vertices, compartment, radius = item
            count += 1
            pbar.update()
    print("count", count)

def main_converted():
    print("main converted")
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    data = dataset.ProofConvertedDataset(config)
    loader = dataset.build_dataloader(data)
    print("size of data", data.__len__())
    with tqdm(total=data.__len__()) as pbar:
        count = 0
        for item in loader:
            root, vertices, edges, rank_0, rank_1, rank_2, num_vertices, num_initial_vertices, compartment, radius = item
            count += 1
            pbar.update()
    print("count", count)

def main_in_memory():
    print("main in memory converted")
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    data = dataset.ProofInMemoryDataset(config)
    loader = dataset.build_dataloader(data)
    print("size of data", data.__len__())
    with tqdm(total=data.__len__()) as pbar:
        count = 0
        for item in loader:
            root, vertices, edges, rank_0, rank_1, rank_2, num_vertices, num_initial_vertices, compartment, radius = item
            count += 1
            pbar.update()
    print("count", count)
if __name__ == "__main__":
    # main()
    main_converted()
    # main_in_memory()