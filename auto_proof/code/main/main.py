from auto_proof.code.model import create_model
from auto_proof.code.train import Trainer
from auto_proof.code.dataset import AutoProofDataset
from auto_proof.code.pre import data_utils

import json
import neptune
import argparse
import torch

def main():
    config = data_utils.get_config()
    ram_alloc = 16
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_workers", help="num workers")
    parser.add_argument("-r", "--ram", help="ram")
    args = parser.parse_args()
    if args.num_workers:
        config['loader']['num_workers'] = int(args.num_workers)
    if args.ram:
        ram_alloc = int(args.ram) 

    fov = config['loader']['fov']
    gpu = str(torch.cuda.get_device_name(0))

    # Create a Neptune run
    run = neptune.init_run(
        project="shawns28/AutoProof", 
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOTA3ZDNjNS0wNGI5LTQ5OWEtYjRkYi05NmFlMzNjNzBkMGIifQ==", 
        name="fill in", 
        tags=[f'{ram_alloc}gb', f'{fov}fov', gpu], 
        dependencies="infer",
        monitoring_namespace="monitoring",
        source_files=["auto_proof/code/model.py", "auto_proof/code/main/main.py", "auto_proof/code/dataset.py", "auto_proof/code/train.py", "auto_proof/code/visualize.py", "auto_proof/code/object_detection.py", "auto_proof/configs/base_config.json"],
    )

    recall_targets = config["trainer"]["recall_targets"]
    thresholds = config["trainer"]["thresholds"]
    config["trainer"]["thresholds"] = str(thresholds)
    config["trainer"]["recall_targets"] = str(recall_targets)
    run["parameters"] = config
    config["trainer"]["thresholds"] = thresholds
    config["trainer"]["recall_targets"] = recall_targets

    print("Tatal num workers", config['loader']['num_workers'])
    print("Batch size" , config['loader']['batch_size'])
    
    print("Building datasets")
    train_dataset = AutoProofDataset(config, 'train')
    val_dataset = AutoProofDataset(config, 'val')
    test_dataset = AutoProofDataset(config, 'test')

    print("Creating model")
    model = create_model(config)
    trainer = Trainer(config, model, train_dataset, val_dataset, test_dataset, run)

    print("Start training")
    trainer.train()
    print("Done")

    run.stop()

if __name__ == "__main__":
   #  multiprocessing.set_start_method('spawn')  # Force 'spawn' This makes it go incredibly slow
    main()
