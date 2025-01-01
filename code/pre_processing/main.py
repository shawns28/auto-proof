import torch
import pickle
import numpy as np
from tqdm import tqdm
import glob
import h5py
import dataset
import json
from dataset import AutoProofDataset, build_dataloader
from model import create_model
from train import Trainer

def test_main():
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    data = AutoProofDataset(config)
    train_loader, val_loader =build_dataloader(data, config)
    print("size of data", data.__len__())
    with tqdm(total=data.__len__() / config['loader']['batch_size']) as pbar:
        for item in train_loader:
            pbar.update()

def main():
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    data = AutoProofDataset(config)
    train_loader, val_loader, train_size, val_size = build_dataloader(data, config)

    model = create_model(config)
    trainer = Trainer(config, model, [train_loader, val_loader], [train_size, val_size])

    print("Start training")
    trainer.train()
    print("Done")

if __name__ == "__main__":
    main()
