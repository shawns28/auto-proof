from auto_proof.code.model import create_model
from auto_proof.code.train import Trainer
from auto_proof.code.dataset import AutoProofDataset

import json
import neptune

CONFIG_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/base_config.json'

def main():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Create a Neptune run
    run = neptune.init_run(
        project="shawns28/AutoProof", 
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOTA3ZDNjNS0wNGI5LTQ5OWEtYjRkYi05NmFlMzNjNzBkMGIifQ==", 
        name="testing_original_features", 
        tags=["sharding", "ac_attention"], 
        dependencies="infer", 
        monitoring_namespace="monitoring",
        source_files=["auto_proof/code/model.py", "auto_proof/code/main/main.py", "auto_proof/code/dataset.py", "auto_proof/code/train.py", "auto_proof/code/visualize.py"],
    )

    run["parameters"] = config
    
    dataset = AutoProofDataset(config)

    model = create_model(config)
    trainer = Trainer(config, model, dataset, run)

    print("Start training")
    trainer.train()
    print("Done")

    run.stop()

if __name__ == "__main__":
    main()
