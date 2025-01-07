from auto_proof.code.model import create_model
from auto_proof.code.train import Trainer

import torch
import pickle
import numpy as np
from tqdm import tqdm
import glob
import h5py
import dataset
import json
from dataset import AutoProofDataset, build_dataloader

CONFIG_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/base_config.json'

def main():
    with open(CONFIG_PATH, 'r') as f:
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
