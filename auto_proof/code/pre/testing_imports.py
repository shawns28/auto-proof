# from ...random import model2 as model
from auto_proof.code.pre import model
import json

CONFIG_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/random/base_config.json'

# Create model
hi = model.GraphTransformer(
                dim=1, 
                depth=1, 
                num_heads=1,
                feat_dim=1,
                num_classes=1)
print("done")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
print(config["data"]["data_path"])
# with open('../configs/base_config.json', 'r') as f:
#         config = json.load(f)
print("loaded json")