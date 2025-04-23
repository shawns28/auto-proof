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
    if not os.path.exists(segclr_dir):
        os.makedirs(segclr_dir)
    if not os.path.exists(roots_at_segclr_dir):
        os.makedirs(roots_at_segclr_dir)

    # roots = ['864691135591041291_000'] # proofread at 943
    # roots = ['864691135463333789_000'] 
    # roots = ['864691135991989185_000'] # conf error missed
    # roots = ['864691136952448223_000'] # another one that has maybe close by branch
    # roots = ['864691134365445988_000'] # not in segclr
     #roots = ['864691137021374830_000'] # in segclr and between 943-1300
    # roots = ['864691134541929231_000'] # not in segclr
    # roots = ['864691134474526022_000'] # not in segclr
    # roots = ['864691134001592490_000'] # not in segclr
    # roots = ['864691134940888163_000'] # The index 3154 is out of bounds for axis 0 with size 3154
    # roots = ['864691134989083258_000'] # Same as above
    # roots = ['864691135133124000_000'] # eror: Bbox([np.int32(123921), np.int32(58980), np.int32(22466)],[np.int32(124049), np.int32(59108), np.int32(22498)], dtype=np.int32, unit='vx')
    roots = data_utils.load_txt(f'{roots_dir}{data_config['features']['post_feature_roots']}')
    print("roots original", len(roots))

    roots_1300_unique_copied = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/1300_unique_copied.txt")
    roots = np.setdiff1d(roots, roots_1300_unique_copied)
    print("roots after removing 1300", len(roots))

    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)
    print("chunk len", len(roots))

    # roots = ['864691137198915137_000', '864691135778235581_000', '864691135463333789_000']
    # roots = ['864691136020979704_000', '864691136273631245_000']

    args_list = list([(root, data_config) for root in roots])
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root, args_list):
            pbar.update()

    # Save all the roots that were successful
    files = glob.glob(f'{segclr_dir}*')
    roots = [files[i][-27:-5] for i in range(len(files))]
    data_utils.save_txt(f'{roots_dir}{data_config['segclr']['post_segclr_roots']}', roots)

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
    if segmentation_version == 1300:
        print("root", root)
    # print("segmentation version", segmentation_version)
    seg_path = data_config['segmentation'][f'precomputed_{segmentation_version}']
    cv_seg = CloudVolume(seg_path, use_https=True)
    resolution = np.array(data_config['segmentation']['resolution'])
    
    status, e, root_at_arr = get_roots_at(feature_path, cv_seg, resolution)
    # print("root at arr", root_at_arr)
    # with h5py.File(f'{data_config['data_dir']}{data_config['labels']['roots_at_latest_dir']}{1300}/{root}.hdf5', 'r') as f:
    #     print(f['roots_at'][:])
    if status == False:
        print("Failed to get roots at for root", root, "eror:", e)
        root_without_iden = root[:-4]
        error_file_path = f"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_at_segclr_error_roots/{root_without_iden}"
        with open(error_file_path, 'w') as file:
            pass
        return

    filesystem = gcsfs.GCSFileSystem(token='anon')
    num_shards = data_config['segclr'][f'num_shards_{segmentation_version}']
    bytewidth = data_config['segclr'][f'bytewidth_{segmentation_version}']
    url = data_config['segclr'][f'url_{segmentation_version}']
    embedding_reader = EmbeddingReader(filesystem, url, sharder, num_shards, bytewidth)
    emb_dim = data_config['segclr']['emb_dim']
    visualize_radius = data_config['segclr']['visualize_radius']
    small_radius = data_config['segclr']['small_radius']
    large_radius = data_config['segclr']['large_radius']

    status, e, segclr_emb, has_emb = get_segclr_emb(root, feature_path, root_at_arr, embedding_reader, emb_dim, visualize_radius, small_radius, large_radius)
    if status == False:
        print("Failed to segclr for root", root, "eror:", e)
        root_without_iden = root[:-4]
        error_file_path = f"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/segclr_error_roots/{root_without_iden}"
        with open(error_file_path, 'w') as file:
            pass
        return
    
    segclr_path = f'{data_dir}{data_config['segclr']['segclr_dir']}{root}.hdf5'
    roots_at_segclr_path = f'{data_dir}{data_config['segclr']['roots_at_segclr_dir']}{root}.hdf5'
    try:
        with h5py.File(roots_at_segclr_path, 'a') as roots_at_segclr_f, h5py.File(segclr_path, 'a') as segclr_f:
            roots_at_segclr_f.create_dataset('roots_at', data=root_at_arr)
            segclr_f.create_dataset('segclr', data=segclr_emb)
            segclr_f.create_dataset('has_emb', data=has_emb)
    except Exception as e:
        print("Failed to save files for root", root, "error:", e)
        error_file_path = f"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/segclr_file_error_roots/{root}"
        with open(error_file_path, 'w') as file:
            pass

def get_roots_at_seglcr_version(root, data_dir, mat_versions):
    """TODO: Fil in and mention the mat versions needs to be 3
    
    """
    mat_version1 = mat_versions[0]
    mat_version2 = mat_versions[1]
    mat_version3 = mat_versions[2]
    roots1 = data_utils.load_txt(f'{data_dir}roots_{mat_version1}_{mat_version2}/post_edit_roots.txt')
    roots2 = data_utils.load_txt(f'{data_dir}proofread/{mat_version2}_unique.txt')
    roots3 = data_utils.load_txt(f'{data_dir}roots_{mat_version2}_{mat_version3}/post_edit_roots.txt')
    roots4 = data_utils.load_txt(f'{data_dir}proofread/{mat_version3}_unique.txt')

    root_without_ident = root[:-4]
    if root_without_ident in roots1:
        segmentation_version = mat_version1
    elif root_without_ident in roots2 or root_without_ident in roots3:
        segmentation_version = mat_version2
    elif root_without_ident in roots4:
        segmentation_version = mat_version3
    else:
        raise Exception("Root not in any of the root lists")
    return segmentation_version

if __name__ == "__main__":
    main()