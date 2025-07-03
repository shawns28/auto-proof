from auto_proof.code.pre import data_utils
from auto_proof.code.pre.roots_at import get_roots_at
from auto_proof.code.connectomics.reader import EmbeddingReader
from auto_proof.code.pre.segclr import get_segclr_emb, sharder, get_roots_at_seglcr_version

import os
import numpy as np
from tqdm import tqdm
import h5py
import time
import multiprocessing
from cloudvolume import CloudVolume
import glob
import gcsfs
import time

def main():
    """Orchestrates the process of obtaining SEGCLR embeddings for neuron roots.

    Steps involved:
    1.  Loads configuration settings and roots
    2.  Removes roots from materialization version 1300, as SEGCLR embeddings are not available for it.
    3.  Processes a chunk of the filtered list of roots using parallel processing.
    4.  Saves the updated list of successfully processed roots to a text file.
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
    
    os.makedirs(segclr_dir, exist_ok=True)
    os.makedirs(roots_at_segclr_dir, exist_ok=True)
    
    roots = data_utils.load_txt(f'{roots_dir}{data_config['features']['post_feature_roots']}')
    print("roots original", len(roots))

    # NOTE: This is due to SegCLR not existing for 1300
    roots_1300_unique_copied = f'{data_dir}{data_config['proofread']['proofread_dir']}1300_unique_copied.txt'
    roots = np.setdiff1d(roots, roots_1300_unique_copied)
    print("roots after removing 1300", len(roots))

    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)
    print("chunk len", len(roots))

    args_list = list([(root, data_config) for root in roots])
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root, args_list):
            pbar.update()

    # Save all the roots that were successful
    files = glob.glob(f'{segclr_dir}*')
    roots = [files[i][-27:-5] for i in range(len(files))]
    data_utils.save_txt(f'{roots_dir}{data_config['segclr']['post_segclr_roots']}', roots)

def process_root(data):
    """Processes a single neuron root to obtain and save its SegCLR embeddings.

    This function handles the sequential steps for a single neuron root:
    1.  Determines output file paths for SegCLR embeddings and roots-at data.
    2.  Skips processing if output files already exist (idempotency).
    3.  Determines the correct segmentation version for the root, handling
        cases where SegCLR embeddings are unavailable for specific versions (e.g., 1300).
    4.  Retrieves the root IDs at the SegCLR version for skeleton vertices from CloudVolume.
    5.  Initializes an `EmbeddingReader` to fetch SegCLR embeddings.
    6.  Obtains and processes SegCLR embeddings for the skeleton vertices.
    7.  Saves the roots-at data, SegCLR embeddings, 
        and if a node has SegCLR embeddings to HDF5 files.

    Args:
        data: A tuple containing:
            - root (str): The unique string identifier for the neuron root.
            - data_config (Dict[str, Any]): A dictionary of global configuration
              parameters.
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
    segmentation_version = get_roots_at_seglcr_version(root, data_dir, mat_versions, False)
    if segmentation_version == 1300:
        print("Got a root at 1300 version. SegCLR embeddings don't exist for 1300 currently.", root)
        return
    seg_path = data_config['segmentation'][f'precomputed_{segmentation_version}']
    cv_seg = CloudVolume(seg_path, use_https=True)
    resolution = np.array(data_config['segmentation']['resolution'])
    
    with h5py.File(feature_path, 'r') as f:
        vertices = f['vertices'][:]
        status, e, root_at_arr = get_roots_at(vertices, cv_seg, resolution)
        if status == False:
            print("Failed to get roots at for root", root, "eror:", e)
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

    with h5py.File(feature_path, 'r') as f:
        vertices = f['vertices'][:]
        edges = f['edges'][:]
        status, e, segclr_emb, has_emb = get_segclr_emb(root, vertices, edges, root_at_arr, embedding_reader, emb_dim, visualize_radius, small_radius, large_radius)
        if status == False:
            print("Failed to segclr for root", root, "eror:", e)
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

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time # Calculate elapsed time
    print(f"\nProcess completed in {elapsed_time:.2f} seconds.")