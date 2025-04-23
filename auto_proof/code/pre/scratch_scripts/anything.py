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

files = glob.glob(f'{"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/segclr/"}*')
roots = [files[i][-27:-5] for i in range(len(files))]
data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/post_segclr_roots.txt", roots)

# current_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/all_roots_461023.txt")
# proofread_943_changed = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_1300_changed.txt")

# result = np.setdiff1d(current_roots, proofread_943_changed)
# print(len(current_roots))
# print(len(proofread_943_changed))
# print(len(result))

# proofread = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_1300.txt")
# result = np.setdiff1d(result, proofread)
# print(len(result))

# result = [str(root) + '_000' for root in result]

# new_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/post_proofread_roots.txt")
# new_root_diff = np.setdiff1d(new_roots, result)
# print(len(new_roots))
# print(len(new_root_diff))
# data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/roots_diff_from_curr.txt", new_root_diff)

# roots_to_remove = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/roots_diff_from_curr.txt")
# print(len(roots_to_remove))
# files = glob.glob(f'{"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/features/"}*')
# roots = [files[i][-27:-5] for i in range(len(files))]
# print(len(roots))
# data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/post_feature_roots.txt", roots)
# roots_diff = np.setdiff1d(roots_to_remove, roots)
# print(len(roots_diff))
# roots_diff = np.intersect1d(roots_diff, roots_to_remove)
# print(len(roots_diff))
# data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/roots_diff_from_curr_2.txt", roots_diff)

# with tqdm(total=len(roots_to_remove)) as pbar:
#     for root in roots_to_remove:
#         file_path = f"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_at_1300/{root}.hdf5"
#         if os.path.exists(file_path):
#             os.remove(file_path)

#         file_path = f"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/labels_at_1300/{root}.hdf5"
#         if os.path.exists(file_path):
#             os.remove(file_path)

#         # file_path = f"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/segclr/{root}.hdf5"
#         # if os.path.exists(file_path):
#         #     os.remove(file_path)
#         # else:
#         #     # print("File path doesn't exist for:", file_path)
#         #     pass
#         # file_path = f"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_at_segclr/{root}.hdf5"
#         # if os.path.exists(file_path):
#         #     os.remove(file_path)
#         # else:
#         #     # print("File path doesn't exist for:", file_path)
#         #     pass
#         pbar.update()



# files = glob.glob(f'{"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/error_roots/"}*')
# roots = [files[i][-18:] for i in range(len(files))]
# data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/feature_error_roots.txt", roots)

# proofread_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_1300_with_copies.txt")
# proofread_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_unique_copied.txt")
# print(len(proofread_roots))
# obj_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/conf_no_error_in_box_roots_val.txt")
# print(len(obj_roots))
# post_label_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/post_label_roots.txt")
# print(len(post_label_roots))
# result = np.setdiff1d(obj_roots, post_label_roots)
# print(len(result))

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