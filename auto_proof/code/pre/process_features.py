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

def main():
    """Processes skeleton features for a batch of roots using multiprocessing.

    This function orchestrates the generation and processing of features for
    individual neuronal skeletons. It divides the roots into chunks for parallel
    processing and saves the extracted features into HDF5 files.

    Assumes that the 'post_generate_roots' file exists and contains the list of
    roots (with identifiers) for which skeletons are expected to be available
    in the CAVE cache.

    Steps involved:
    1. Loads configuration settings for data and client, and multiprocessing parameters.
    2. Initializes the CAVE client.
    3. Sets up necessary directories for storing feature files.
    4. Loads mappings for representative edit coordinates and the list of proofread roots.
    5. Loads the list of roots that have successfully had their skeletons generated/verified.
    6. Chunks the roots based on the provided multiprocessing parameters.
    7. Prepares arguments for each root, determining if a representative coordinate
       is needed (for raw edit roots) or not (for proofread roots).
    8. Uses a multiprocessing pool to process root features in parallel, displaying
       progress with a tqdm bar.
    9. After processing, updates the list of successfully processed roots and saves
       it to a text file.
    """
    data_config = data_utils.get_config('data')
    client_config = data_utils.get_config('client')
    data_config, chunk_num, num_chunks, num_processes = data_utils.get_num_chunk_and_processes(data_config)

    data_dir = data_config['data_dir']
    client, datastack_name, mat_version_start, mat_version_end = data_utils.create_client(client_config)

    roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'
    dicts_dir = f'{data_dir}dicts_{mat_version_start}_{mat_version_end}/'
    features_dir = f'{data_dir}{data_config['features']['features_dir']}'
    os.makedirs(features_dir, exist_ok=True)

    root_to_rep_coord = data_utils.load_pickle_dict(f'{dicts_dir}{data_config['raw_edits']['root_to_rep']}')

    skeleton_version = data_config['features']['skeleton_version']

    post_generate_roots_path = f'{roots_dir}{data_config['generate_skels']['post_generate_roots']}'
    roots = data_utils.load_txt(post_generate_roots_path)
    print("roots len", len(roots))
    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)
    
    args_list = []
    for root in roots:
        root_without_identifier = int(root[:-4])
        if root_without_identifier not in root_to_rep_coord:
            rep = None
        else:
            rep = root_to_rep_coord[root_without_identifier]
        args_list.append((root, data_config, rep, client, datastack_name))

    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root_features, args_list):
            pbar.update()

    files = glob.glob(f'{features_dir}*')
    roots = [files[i][-27:-5] for i in range(len(files))]
    data_utils.save_txt(f'{roots_dir}{data_config['features']['post_feature_roots']}', roots)

def process_root_features(data):
    """Processes features for a single root ID.

    This helper function is designed to be run in parallel by multiprocessing.
    It performs the following steps for a given root:
    1. Checks if the HDF5 feature file for this root already exists to skip reprocessing.
    2. Retrieves the skeleton from the CAVE client.
    3. Processes the raw skeleton data into a structured feature dictionary.
    4. Applies positional encoding (map_pe) to the graph edges.
    5. Saves all extracted features into an HDF5 file.

    Args:
        data: A tuple containing the following elements:
            - root (str): The root ID with identifier (e.g., '12345_000').
            - data_config (dict): Data configuration settings.
            - rep (np.ndarray or None): Representative edit coordinate for the root,
              or None if it's a proofread root.
            - client (Any): The CAVE client object.
            - datastack_name (str): The name of the datastack.
            - skeleton_version (str): The skeleton version to use.
    """
    root, data_config, rep, client, datastack_name = data

    feature_path = f'{data_config['data_dir']}{data_config['features']['features_dir']}{root}.hdf5'
    # Skip already processed roots
    if os.path.exists(feature_path):
        return 

    # Skeletonize
    status, e, skel_dict = get_skel(datastack_name, data_config['features']['skeleton_version'], root, client)
    if status == False:
        print("Failed to get skel for root", root, "eror:", e)
        return
    status, e, feature_dict = process_skel(data_config['features']['box_cutoff'], data_config['features']['cutoff'], rep, skel_dict)
    if len(feature_dict['vertices']) == 1:
        print("Root had one vertice, skipping", root)
        return
    if status == False:
        print("Failed to process for root", root, "eror:", e)
        return
    
    # map pe
    status, e, map_pe = map_pe_wrapper(data_config['features']['pos_enc_dim'], feature_dict['edges'], len(feature_dict['vertices']))
    if status == False:
        print("Failed map pe for root", root, "eror:", e)
        return
    feature_dict['map_pe'] = map_pe

    with h5py.File(feature_path, 'a') as feat_f:
        for feature in feature_dict:
            feat_f.create_dataset(feature, data=feature_dict[feature])

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nProcess completed in {elapsed_time:.2f} seconds.")