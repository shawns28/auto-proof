from auto_proof.code.pre import data_utils
from auto_proof.code.pre.skeletonize import get_skel, process_skel
from auto_proof.code.pre.map_pe import map_pe_wrapper

import os
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import glob
import time


# proofread_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_1300_with_copies.txt")
proofread_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_unique_copied.txt")
print(len(proofread_roots))
post_label_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/post_label_roots.txt")
print(len(post_label_roots))
result = np.intersect1d(proofread_roots, post_label_roots)
print(len(result))

# path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_unique.txt"
# roots_943 = data_utils.load_txt(path)
# print(len(roots_943))

# roots_343_943 = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_943/post_edit_roots.txt")
# print(len(roots_343_943))

# roots_943_1300 = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_943_1300/post_edit_roots.txt")
# print(len(roots_943_1300))

# roots_343_1300 = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/post_edit_roots.txt")
# print(len(roots_343_1300))

# print("results")
# result = np.intersect1d(roots_943, roots_343_943)
# print(len(result))

# result2 = np.setdiff1d(roots_343_1300, roots_943_1300)
# print(len(result2))
# result3 = np.intersect1d(result2, roots_343_943)
# print(len(result3)) 

# result4 = np.intersect1d(roots_943, roots_343_1300)
# print(len(result4))


# Check full list and see if 343-943 + 943 + 943-1300 + 1300 adds up to everything merged together