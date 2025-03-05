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
