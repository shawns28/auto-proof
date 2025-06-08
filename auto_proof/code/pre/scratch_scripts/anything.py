from auto_proof.code.pre import data_utils

import os
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import glob
import time

# missed_one = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/before_sven/missed_0.05thres_0.1cloud_valincluding.txt")
found_two = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/before_sven/found_0.05thres_0.1cloud_val.txt")
missed_segclr = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/before_sven/missed_nosegclr.txt")
print(len(found_two))
print(len(missed_segclr))
intersect = np.intersect1d(found_two, missed_segclr)
# intersect = np.setdiff1d(missed_one, missed_two)
print(len(intersect))
data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/before_sven/intersect_nosegclr.txt", intersect)

# files = glob.glob(f'{"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/segclr/"}*')
# roots = [files[i][-27:-5] for i in range(len(files))]
# data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/post_segclr_roots.txt", roots)

# current_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/all_roots_461023.txt")
# # proofread_943_changed = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_1300_changed.txt")

# # result = np.setdiff1d(current_roots, proofread_943_changed)
# # print(len(current_roots))
# # print(len(proofread_943_changed))
# result = current_roots

# result = [str(root) + '_000' for root in result]

# data_config = data_utils.get_config('data')
# # client_config = data_utils.get_config('client')

# data_dir = data_config['data_dir']
# mat_version_start = 343
# mat_version_end = 1300

# roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'
# features_dir = f'{data_dir}{data_config['features']['features_dir']}'
# labels_dir = f'{data_dir}{data_config['labels']['labels_at_latest_dir']}{data_config['labels']['latest_mat_version']}/'

# post_label_roots = data_utils.load_txt(f'{roots_dir}{data_config['labels']['post_label_roots']}')
# print("post_label_roots len", len(post_label_roots))

# post_segclr_roots = data_utils.load_txt(f'{roots_dir}{data_config['segclr']['post_segclr_roots']}')
# print("post_segclr_roots len", len(post_segclr_roots))
# roots = np.intersect1d(post_label_roots, post_segclr_roots)
# print("roots combined len", len(roots))
# roots_1300_unique_copied = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/1300_unique_copied.txt")
# roots = np.setdiff1d(roots, roots_1300_unique_copied)
# print("final roots len", len(roots))

# result2 = np.intersect1d(result, roots)
# print("final result2", len(result2))
# data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/new_and_prev_shared_roots.txt", result2)

# print("hi")

# files = glob.glob(f'{"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/segclr/"}*')
# roots = [files[i][-27:-5] for i in range(len(files))]
# proofread = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_unique.txt")
# proofread = [str(root) + '_000' for root in proofread]
# print(len(proofread))
# result = np.intersect1d(roots, proofread)
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