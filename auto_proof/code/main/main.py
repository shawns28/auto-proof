from auto_proof.code.model import create_model
from auto_proof.code.train import Trainer
from auto_proof.code.dataset import AutoProofDataset
from auto_proof.code.pre import data_utils

import json
import neptune
import argparse
import multiprocessing
import torch

def main():
    config = data_utils.get_config()
    ram = 32
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_workers", help="num workers")
    parser.add_argument("-f", "--fov", help="fov")
    parser.add_argument("-r", "--ram", help="ram")
    args = parser.parse_args()
    if args.num_workers:
        config['loader']['num_workers'] = int(args.num_workers)
    if args.fov:
        config['loader']['fov'] = int(args.fov)
    if args.ram:
        ram = int(args.ram) 

    fov = config['loader']['fov']
    gpu = torch.cuda.get_device_name(0)

    # Create a Neptune run
    run = neptune.init_run(
        project="shawns28/AutoProof", 
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOTA3ZDNjNS0wNGI5LTQ5OWEtYjRkYi05NmFlMzNjNzBkMGIifQ==", 
        name="fill in", 
        tags=[f"{fov}fov", f"{ram}gb", gpu], 
        dependencies="infer",
        monitoring_namespace="monitoring",
        source_files=["auto_proof/code/model.py", "auto_proof/code/main/main.py", "auto_proof/code/dataset.py", "auto_proof/code/train.py", "auto_proof/code/visualize.py"],
    )

    run["parameters"] = config
    
    train_dataset = AutoProofDataset(config, 'train')
    val_dataset = AutoProofDataset(config, 'val')
    test_dataset = AutoProofDataset(config, 'test')

    model = create_model(config)
    trainer = Trainer(config, model, train_dataset, val_dataset, test_dataset, run)

    print("Start training")
    trainer.train()
    print("Done")

    run.stop()

if __name__ == "__main__":
   #  multiprocessing.set_start_method('spawn')  # Force 'spawn' This makes it go incredibly slow
    main()
