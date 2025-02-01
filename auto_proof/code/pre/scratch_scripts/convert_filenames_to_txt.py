from auto_proof.code.pre import data_utils

import numpy as np
import glob

SUCCESSFUL_ROOTS_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/successful_sharded_roots_features_conf/'
TXT_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/features_conf.txt'

# Converting all the filenames which represent names of each root to a txt with all the roots in them
files = glob.glob(f'{SUCCESSFUL_ROOTS_PATH}/*')
roots = [files[i][-18:] for i in range(len(files))]
print("first root", roots[0])
data_utils.save_txt(TXT_PATH, roots)

# Comparing the roots after the operation to before
before_op = data_utils.load_txt('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/post_label_roots_459972.txt')
after_op = data_utils.load_txt('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/features_conf.txt')
diff = np.setdiff1d(before_op, after_op)
print("different: ", len(diff))
if len(diff) > 0:
    data_utils.save_txt(f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/missing_shard_roots_features_conf_{len(diff)}.txt', diff)
