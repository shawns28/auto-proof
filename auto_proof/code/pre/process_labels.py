from auto_proof.code.pre import data_utils
from auto_proof.code.pre.roots_at import get_roots_at
from auto_proof.code.pre.labels import create_labels
from auto_proof.code.pre.distance import create_dist

import os
import numpy as np
from tqdm import tqdm
import h5py
import time
import multiprocessing
from cloudvolume import CloudVolume
import glob

def main():
    """Orchestrates the feature labeling and distance calculation pipeline for neuron roots.

    Creates the labels, confidences and distances for each root.

    If labels_type == ignore_inbetween, ignore_edge_ccs should be False.
    If labels_type == ignore_inbetween_and_edge, ignore_edge_ccs should be True.
    labels_type == ignore_nothing is deprecated but represented the original labels.
    
    Steps involved:
    1.  Does setup configuration steps.
    2.  Initializes a multiprocessing pool to process roots in parallel, displaying progress.
    3.  After all roots in the current chunk are processed, identifies roots for which
        output files were successfully generated.
    4.  Saves the updated list of successfully processed roots to a text file.
    """
    data_config = data_utils.get_config('data')
    client_config = data_utils.get_config('client')
    data_config, chunk_num, num_chunks, num_processes = data_utils.get_num_chunk_and_processes(data_config)

    data_dir = data_config['data_dir']
    mat_version_start = client_config['client']['mat_version_start']
    mat_version_end = client_config['client']['mat_version_end']
    roots_dir = f'{data_config['data_dir']}roots_{mat_version_start}_{mat_version_end}/'
    latest_version = data_config['labels']['latest_mat_version']
    labels_at_latest_dir = f'{data_dir}{data_config['labels']['labels_at_latest_dir']}' \
                 f'{latest_version}_' \
                 f'{data_config['labels']['labels_type']}/'
    roots_at_latest_dir = f'{data_dir}{data_config['labels']['roots_at_latest_dir']}{latest_version}/'
    os.makedirs(labels_at_latest_dir, exist_ok=True)
    os.makedirs(roots_at_latest_dir,  exist_ok=True)

    roots = data_utils.load_txt(f'{roots_dir}{data_config['features']['post_feature_roots']}')
    
    print("roots len", len(roots))
    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)
    print("chunk len", len(roots))

    args_list = list([(root, data_config) for root in roots])
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root, args_list):
            pbar.update()

    # Save all the roots that were successful
    files = glob.glob(f'{labels_at_latest_dir}*')
    roots = [files[i][-27:-5] for i in range(len(files))]
    data_utils.save_txt(f'{roots_dir}{data_config['labels']['post_label_roots']}', roots)

def process_root(data):
    """Processes a single neuron root to generate its labels, confidences, and error distances.

    This function handles the sequential steps for a single neuron root:
    1. Determines output file paths for labels and roots-at data.
    2. Skips processing if output files already exist (idempotency).
    3. Retrieves segment IDs (roots-at) for skeleton vertices using CloudVolume,
       caching results for future runs.
    4. Computes error labels and associated confidences based on root-at data
       and known proofread roots.
    5. Calculates the shortest distance from each vertex to any identified error vertex.
    6. Saves all generated data into HDF5 files.

    Args:
        data: A tuple containing:
            - root (str): The unique string identifier for the neuron root.
            - data_config (Dict[str, Any]): A dictionary of global configuration
              parameters.
    """
    root, data_config = data

    latest_version = data_config['labels']['latest_mat_version']
    labels_at_latest_path = f'{data_config['data_dir']}{data_config['labels']['labels_at_latest_dir']}' \
                 f'{latest_version}_' \
                 f'{data_config['labels']['labels_type']}/' \
                 f'{root}.hdf5'
    roots_at_latest_path = f'{data_config['data_dir']}{data_config['labels']['roots_at_latest_dir']}{latest_version}/{root}.hdf5'
    # Skip already processed roots
    if os.path.exists(labels_at_latest_path) and os.path.exists(roots_at_latest_path):
        return

    feature_path = f'{data_config['data_dir']}{data_config['features']['features_dir']}{root}.hdf5'
    
    # root at latest
    if not os.path.exists(roots_at_latest_path):
        seg_path = data_config['segmentation'][f'precomputed_{latest_version}']
        cv_seg = CloudVolume(seg_path, use_https=True)
        resolution = np.array(data_config['segmentation']['resolution'])
        with h5py.File(feature_path, 'r') as f:
            vertices = f['vertices'][:]
            status, e, root_at_arr = get_roots_at(vertices, cv_seg, resolution)
            if status == False:
                print("Failed to get roots at for root", root, "eror:", e)
                return
    else:
        with h5py.File(roots_at_latest_path, 'r') as roots_at_f:
            root_at_arr = roots_at_f['roots_at'][:]

    # labels
    proofread_mat_version1 = data_config['proofread']['mat_versions'][0]
    proofread_mat_version2 = data_config['proofread']['mat_versions'][1]
    proofread_roots_path = f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{proofread_mat_version1}_{proofread_mat_version2}.txt'
    proofread_roots = data_utils.load_txt(proofread_roots_path)
    ignore_edge_ccs = data_config['labels']['ignore_edge_ccs']
    with h5py.File(feature_path, 'r') as f:
        edges = f['edges'][:]
    labels, confidences = create_labels(root_at_arr, ignore_edge_ccs, edges, proofread_roots)

    # distance to error
    with h5py.File(feature_path, 'r') as f:
        edges = f['edges'][:]
        dist = create_dist(edges, labels)

    # Save roots at and labels, confidences, dist
    if not os.path.exists(roots_at_latest_path):
        with h5py.File(roots_at_latest_path, 'a') as roots_at_f:
            roots_at_f.create_dataset('roots_at', data=root_at_arr)
    with h5py.File(labels_at_latest_path, 'a') as labels_f:
        labels_f.create_dataset('labels', data=labels)
        labels_f.create_dataset('confidences', data=confidences)
        labels_f.create_dataset('dist', data=dist)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time # Calculate elapsed time
    print(f"\nProcess completed in {elapsed_time:.2f} seconds.")