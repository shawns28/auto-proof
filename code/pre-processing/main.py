import torch
import pickle
import numpy as np
from tqdm import tqdm
import glob
import h5py
import dataset
import json

def main():
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    data = dataset.AutoProofDataset(config)
    loader = dataset.build_dataloader(data)
    print("size of data", data.__len__())
    with tqdm(total=data.__len__()) as pbar:
        for item in loader:
            pbar.update()

if __name__ == "__main__":
    main()
