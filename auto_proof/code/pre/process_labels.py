from auto_proof.code.pre import data_utils
from auto_proof.code.pre.roots_at import get_roots_at
from auto_proof.code.pre.labels import create_labels
from auto_proof.code.pre.distance import create_dist

import os
import numpy as np
from tqdm import tqdm
import h5py
import time
import argparse
import multiprocessing
from cloudvolume import CloudVolume
import glob

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
    roots_dir = f'{data_config['data_dir']}roots_{mat_version_start}_{mat_version_end}/'
    latest_version = data_config['labels']['latest_mat_version']
    labels_at_latest_dir = f'{data_dir}{data_config['labels']['labels_at_latest_dir']}{latest_version}/'
    roots_at_latest_dir = f'{data_dir}{data_config['labels']['roots_at_latest_dir']}{latest_version}/'
    if not os.path.exists(labels_at_latest_dir):
        os.makedirs(labels_at_latest_dir)
    if not os.path.exists(roots_at_latest_dir):
        os.makedirs(roots_at_latest_dir)

    # roots = data_utils.load_txt(f'{roots_dir}{data_config['features']['post_feature_roots']}')
    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_unique_copied.txt")
    print("roots len", len(roots))
    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)
    print("chunk len", len(roots))

    # roots = ['864691137198915137_000', '864691135778235581_000', '864691135463333789_000']

    args_list = list([(root, data_config) for root in roots])
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root, args_list):
            pbar.update()

    # Save all the roots that were successful
    files = glob.glob(f'{labels_at_latest_dir}*')
    roots = [files[i][-27:-5] for i in range(len(files))]
    data_utils.save_txt(f'{roots_dir}{data_config['labels']['post_label_roots']}', roots)

def process_root(data):
    """
        TODO: Fill in
    """
    root, data_config = data

    latest_version = data_config['labels']['latest_mat_version']
    labels_at_latest_path = f'{data_config['data_dir']}{data_config['labels']['labels_at_latest_dir']}{latest_version}/{root}.hdf5'
    roots_at_latest_path = f'{data_config['data_dir']}{data_config['labels']['roots_at_latest_dir']}{latest_version}/{root}.hdf5'
    # Skip already processed roots
    if os.path.exists(labels_at_latest_path) and os.path.exists(roots_at_latest_path):
        return

    feature_path = f'{data_config['data_dir']}{data_config['features']['features_dir']}{root}.hdf5'
    
    # root at latest
    seg_path = data_config['segmentation'][f'precomputed_{latest_version}']
    cv_seg = CloudVolume(seg_path, use_https=True)
    resolution = np.array(data_config['segmentation']['resolution'])
    status, e, root_at_arr = get_roots_at(feature_path, cv_seg, resolution)
    if status == False:
        print("Failed to get roots at for root", root, "eror:", e)
        return

    # labels
    proofread_mat_version1 = data_config['proofread']['mat_versions'][0]
    proofread_mat_version2 = data_config['proofread']['mat_versions'][1]
    proofread_roots_path = f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{proofread_mat_version1}_{proofread_mat_version2}.txt'
    labels, confidences = create_labels(root, root_at_arr, feature_path, proofread_roots_path)

    # distance to error
    dist = create_dist(root, feature_path, labels)

    # Save roots at and labels, confidences, dist
    with h5py.File(roots_at_latest_path, 'a') as roots_at_f, h5py.File(labels_at_latest_path, 'a') as labels_f:
        roots_at_f.create_dataset('roots_at', data=root_at_arr)
        labels_f.create_dataset('labels', data=labels)
        labels_f.create_dataset('confidences', data=confidences)
        labels_f.create_dataset('dist', data=dist)

if __name__ == "__main__":
    main()