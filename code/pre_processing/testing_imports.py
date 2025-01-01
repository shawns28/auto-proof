# from auto_proof.code.pre_processing import model
from .... import code.pre_processing.model
# import model
# Create model
hi = model.GraphTransformer(
                dim=1, 
                depth=1, 
                num_heads=1,
                feat_dim=1,
                num_classes=1)
print("done")