from auto_proof.code.pre import data_utils
from auto_proof.code.dataset import AutoProofDataset
from auto_proof.code.dataset import build_dataloader

import numpy as np
from tqdm import tqdm
import h5py
from sklearn.decomposition import IncrementalPCA
import multiprocessing
import torch
import time
import glob
import joblib
import random

def pca_segclr():
    data_config = data_utils.get_config('data')
    num_processes = 64
    n_components = 3 # rbg
    
    # ipca = IncrementalPCA(n_components=n_components)
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/all_roots.txt")
    # print("original roots size", len(roots))

    # sample_size = int(len(roots) * 0.01)
    # roots = np.random.choice(roots, size=sample_size, replace=False)
    # print("sampled roots len", len(roots))
    # data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/sampled_roots_1.txt", roots)
    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/sampled_roots_1.txt")

    # Don't need to scale ahead of time
    # args_list = list([(root, data_config) for root in roots])
    # total_max = -np.inf
    # total_min = np.inf
    # with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
    #     for curr_max, curr_min in pool.imap_unordered(get_max, args_list):
    #         total_max = max(total_max, curr_max)
    #         total_min = min(total_min, curr_min)
    #         pbar.update()
    # print("max", total_max)
    # print("min", total_min)
    # args_list = list([(root, data_config, total_max, total_min) for root in roots])

    # with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
    #     for emb in pool.imap_unordered(process_root, args_list):
    #         ipca.partial_fit(emb)
    #         pbar.update()
    save_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/segclr_pca.joblib"
    # save_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/segclr_scaled_pca.joblib"

    # joblib.dump(ipca, save_path)
    print("loading pca model")
    loaded_ipca = joblib.load(save_path)
    total_max = np.full(n_components, -np.inf)
    total_min = np.full(n_components, np.inf)
    args_list = list([(root, data_config, loaded_ipca) for root in roots])
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
         for curr_max, curr_min in pool.imap_unordered(get_pca_max, args_list):
            total_max = np.maximum(total_max, curr_max)
            total_min = np.minimum(total_min, curr_min)
            pbar.update()
    print("pca max", total_max)
    print("pca min", total_min)
    np.save('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/pca_total_max.npy', total_max)
    np.save('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/pca_total_min.npy', total_min)

    range_vals = total_max - total_min
    range_vals[range_vals == 0] = 1.0 # Avoid division by zero, component will map to 0.5 or 0

    
    root = '864691135682085268_000'
    root = '864691134940888163_000'
    root = '864691135187356435_000' # half that works with segclr emb being added
    root = '864691135572385061_000' # proofread that should have soma in it
    segclr_path = f'{data_config['data_dir']}{data_config['segclr']['segclr_dir']}{root}.hdf5'
    with h5py.File(segclr_path, 'r') as segclr_f:
        np.set_printoptions(threshold=10000)
        segclr = segclr_f['segclr'][:]
        # print("segclr", segclr)
        segclr_rgb = loaded_ipca.transform(segclr)
        # total_min = np.min(segclr_rgb)
        # total_max = np.max(segclr_rgb)
        print("pca segclr", segclr_rgb)
        segclr_scaled = (segclr_rgb - total_min) / range_vals
        segclr_clipped = np.clip(segclr_scaled, 0, 1)
        print(segclr_clipped)

# def process_root(args):
#     root, data_config, total_max, total_min = args
#     segclr_path = f'{data_config['data_dir']}{data_config['segclr']['segclr_dir']}{root}.hdf5'
#     try:
#         with  h5py.File(segclr_path, 'r') as segclr_f:
#             segclr = segclr_f['segclr'][:]
#             # print(segclr.shape)
#             # has_emb = segclr_f['has_emb'][:]
#             segclr_scaled = (segclr - total_min) / (total_max - total_min)
#             return segclr_scaled

#     except Exception as e:
#         print("root: ", root, "error: ", e)
#         return None
    
def process_root(args):
    root, data_config = args
    segclr_path = f'{data_config['data_dir']}{data_config['segclr']['segclr_dir']}{root}.hdf5'
    try:
        with  h5py.File(segclr_path, 'r') as segclr_f:
            segclr = segclr_f['segclr'][:]
            # print(segclr.shape)
            # has_emb = segclr_f['has_emb'][:]
            return segclr

    except Exception as e:
        print("root: ", root, "error: ", e)
        return None
    
def get_max(args):
    root, data_config = args
    segclr_path = f'{data_config['data_dir']}{data_config['segclr']['segclr_dir']}{root}.hdf5'
    try:
        with h5py.File(segclr_path, 'r') as segclr_f:
            segclr = segclr_f['segclr'][:]
            # print(segclr.shape)
            # has_emb = segclr_f['has_emb'][:]
            return np.max(segclr), np.min(segclr)

    except Exception as e:
        print("root: ", root, "error: ", e)
        return None
    
def get_pca_max(args):
    root, data_config, loaded_ipca = args
    segclr_path = f'{data_config['data_dir']}{data_config['segclr']['segclr_dir']}{root}.hdf5'
    try:
        with h5py.File(segclr_path, 'r') as segclr_f:
            segclr = segclr_f['segclr'][:]
            transformed_segclr = loaded_ipca.transform(segclr)
            # transformed_segclr = np.log1p(transformed_segclr)
            # print(segclr.shape)
            # has_emb = segclr_f['has_emb'][:]
            return transformed_segclr.max(axis=0), transformed_segclr.min(axis=0)
    except Exception as e:
        print("root: ", root, "error: ", e)
        return None
if __name__ == "__main__":
    pca_segclr()