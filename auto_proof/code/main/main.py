from auto_proof.code.model import create_model
from auto_proof.code.train import Trainer
from auto_proof.code.dataset import AutoProofDataset
from auto_proof.code.pre import data_utils

import json
import neptune
import argparse
import torch

def main():
    """
    Main function to set up and run the AutoProof model training.

    This function handles:
    1. Parsing command-line arguments for dynamic configuration.
    2. Loading the base configuration.
    3. Initializing Neptune AI for experiment tracking.
    4. Preparing datasets and data loaders.
    5. Creating and initializing the model.
    6. Instantiating and running the Trainer.
    7. Stopping the Neptune run upon completion.
    """
    config = data_utils.get_config('base')
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

    if config['loader']['use_segclr']:
        config['loader']['feat_dim'] = 101
    else:
        config['loader']['feat_dim'] = 36

    # Create a Neptune run
    run = neptune.init_run(
        project="shawns28/AutoProof", 
        api_token="FILL IN", 
        name="fill in optionally", 
        tags=[f'{ram_alloc}gb', f'{fov}fov', gpu], 
        dependencies="infer",
        monitoring_namespace="monitoring",
        source_files=["auto_proof/configs/base_config.json", "auto_proof/code/model.py", "auto_proof/code/main/main.py", "auto_proof/code/dataset.py", "auto_proof/code/train.py", "auto_proof/code/visualize.py", "auto_proof/code/object_detection.py"],
    )

    recall_targets = config["trainer"]["recall_targets"]
    thresholds = config["trainer"]["thresholds"]
    branch_degrees = config["trainer"]["branch_degrees"]
    obj_det_error_cloud_ratios = config["trainer"]["obj_det_error_cloud_ratios"]
    config["trainer"]["thresholds"] = str(thresholds)
    config["trainer"]["recall_targets"] = str(recall_targets)
    config["trainer"]["branch_degrees"] = str(branch_degrees)
    config["trainer"]["obj_det_error_cloud_ratios"] = str(obj_det_error_cloud_ratios)
    run["parameters"] = config
    config["trainer"]["thresholds"] = thresholds
    config["trainer"]["recall_targets"] = recall_targets
    config["trainer"]["branch_degrees"] = branch_degrees
    config["trainer"]["obj_det_error_cloud_ratios"] = obj_det_error_cloud_ratios

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
    main()
