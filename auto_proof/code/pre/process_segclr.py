from auto_proof.code.pre import data_utils
from auto_proof.code.pre.roots_at import get_roots_at
from auto_proof.code.connectomics.reader import EmbeddingReader
from auto_proof.code.connectomics.sharding import md5_shard
from auto_proof.code.pre.segclr import get_segclr_emb, sharder

import os
import numpy as np
from tqdm import tqdm
import h5py
import time
import argparse
import multiprocessing
from cloudvolume import CloudVolume
import glob
import sys
import gcsfs
from scipy.spatial import cKDTree

def main():
    """
        TODO: Fill in
    """
    data_config = data_utils.get_config('data')
    client_config = data_utils.get_config('client')
    data_config, chunk_num, num_chunks, num_processes = data_utils.get_num_chunk_and_processes(data_config)

    data_dir = data_config['data_dir']
    mat_version_start = client_config['client']['mat_version_start']
    mat_version_end = client_config['client']['mat_version_end']
    roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'
    segclr_dir = f'{data_dir}{data_config['segclr']['segclr_dir']}'
    roots_at_segclr_dir = f'{data_dir}{data_config['segclr']['roots_at_segclr_dir']}'

    # roots = ['864691135591041291_000'] # proofread at 943
    roots = ['864691135463333789_000'] 
    # roots = ['/864691135517565322_00']
    # roots = data_utils.load_txt(f'{roots_dir}{data_config['features']['post_feature_roots']}')
    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)
    print("roots len", len(roots))

    # roots = ['864691137198915137_000', '864691135778235581_000', '864691135463333789_000']

    args_list = list([(root, data_config) for root in roots])
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root, args_list):
            pbar.update()

    # Save all the roots that were successful
    # files = glob.glob(f'{segclr_dir}*')
    # roots = [files[i][-27:-5] for i in range(len(files))]
    # data_utils.save_txt(f'{roots_dir}{data_config['segclr']['post_segclr_roots']}', roots)

def process_root(data):
    """
        TODO: Fill in
    """
    root, data_config = data

    data_dir = data_config['data_dir']
    segclr_path = f'{data_dir}{data_config['segclr']['segclr_dir']}{root}.hdf5'
    roots_at_segclr_path = f'{data_dir}{data_config['segclr']['roots_at_segclr_dir']}{root}.hdf5'

    # Skip already processed roots
    if os.path.exists(segclr_path) and os.path.exists(roots_at_segclr_path):
        return

    feature_path = f'{data_config['data_dir']}{data_config['features']['features_dir']}{root}.hdf5'
    
    # root at latest
    mat_versions = data_config['segclr']['mat_versions']
    segmentation_version = get_roots_at_seglcr_version(root, data_dir, mat_versions)
    print("segmentation version", segmentation_version)
    seg_path = data_config['segmentation'][f'precomputed_{segmentation_version}']
    cv_seg = CloudVolume(seg_path, use_https=True)
    resolution = np.array(data_config['segmentation']['resolution'])
    
    status, e, root_at_arr = get_roots_at(feature_path, cv_seg, resolution)
    if status == False:
        print("Failed to get roots at for root", root, "eror:", e)
        return

    filesystem = gcsfs.GCSFileSystem(token='anon')
    num_shards = data_config['segclr'][f'num_shards_{segmentation_version}']
    bytewidth = data_config['segclr'][f'bytewidth_{segmentation_version}']
    url = data_config['segclr'][f'url_{segmentation_version}']
    embedding_reader = EmbeddingReader(filesystem, url, sharder, num_shards, bytewidth)
    emb_dim = data_config['segclr']['emb_dim']

    status, e, segclr_emb = get_segclr_emb(root, feature_path, root_at_arr, embedding_reader, emb_dim)
    
    # Save roots at and labels, confidences, dist
    # with h5py.File(roots_at_latest_path, 'a') as roots_at_f, h5py.File(labels_at_latest_path, 'a') as labels_f:
    #     roots_at_f.create_dataset('roots_at', data=root_at_arr)

def get_roots_at_seglcr_version(root, data_dir, mat_versions):
    """TODO: Fil in and mention the mat versions needs to be 3
    
    """
    mat_version1 = mat_versions[0]
    mat_version2 = mat_versions[1]
    mat_version3 = mat_versions[2]
    roots1 = data_utils.load_txt(f'{data_dir}roots_{mat_version1}_{mat_version2}/post_edit_roots.txt')
    roots2 = data_utils.load_txt(f'{data_dir}proofread/{mat_version2}_unique.txt')
    roots3 = data_utils.load_txt(f'{data_dir}roots_{mat_version2}_{mat_version3}/post_edit_roots.txt')
    
    root_without_ident = root[:-4]
    if root_without_ident in roots1:
        segmentation_version = mat_version1
    elif root_without_ident in roots2:
        segmentation_version = mat_version2
    elif root_without_ident in roots3:
        segmentation_version = mat_version3
    else:
        raise Exception("Root not in any of the root lists")
    return segmentation_version

if __name__ == "__main__":
    main()