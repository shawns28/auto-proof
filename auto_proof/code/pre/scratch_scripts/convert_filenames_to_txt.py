from auto_proof.code.pre import data_utils

import numpy as np
import glob

SUCCESSFUL_ROOTS_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/mixed_isolated_error/'
TXT_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/conf_with_isolated_errors.txt'

# Converting all the filenames which represent names of each root to a txt with all the roots in them
files = glob.glob(f'{SUCCESSFUL_ROOTS_PATH}/*')
roots = [files[i][-22:] for i in range(len(files))]
print("first root", roots[0])
data_utils.save_txt(TXT_PATH, roots)
