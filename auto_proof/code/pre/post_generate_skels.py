from auto_proof.code.pre import data_utils

import os
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import glob
import time

def main():
    """Verifies the existence of generated skeletons and updates the root list.

    It assumes that the input 'post_proofread_roots' file
    already contains the combined list of initial raw edit roots and proofread roots
    with their unique identifiers.

    This function performs the following steps:
    1. Chunks the roots and queries the CAVE client to check if skeletons for
       these roots (without identifiers) exist in the cache for the specified
       skeleton version and datastack.
    2. Creates a boolean mask based on the existence check.
    3. Filters the original list of roots (with identifiers) to retain only
       those for which skeletons exist.
    4. Saves the filtered list of roots to a text file for subsequent steps.
    """
    data_config = data_utils.get_config('data')
    client_config = data_utils.get_config('client')

    data_dir = data_config['data_dir']
    client, datastack_name, mat_version_start, mat_version_end = data_utils.create_client(client_config)

    roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'
    dicts_dir = f'{data_dir}dicts_{mat_version_start}_{mat_version_end}/'

    roots_file_path = f'{roots_dir}{data_config['proofread']['post_proofread_roots']}'
    
    roots = data_utils.load_txt(roots_file_path)
    print(f"Loaded {len(roots)} roots for existence check.")

    post_generate_roots_path = f'{roots_dir}{data_config['generate_skels']['post_generate_roots']}'

    skeleton_version = data_config['features']['skeleton_version']

    exist_chunk_size = data_config['generate_skels']['exists_chunk_size']
    exists_chunks = [roots[i:i + exist_chunk_size] for i in range(0, len(roots), exist_chunk_size)]
    
    # Convert roots with identifiers (e.g., '123_000') to bare root IDs (e.g., 123)
    # as expected by client.skeleton.skeletons_exist
    exists_chunks_without_ident = [[int(root.split('_')[0]) for root in chunk] for chunk in exists_chunks]
    
    print(f"Divided roots into {len(exists_chunks_without_ident)} chunks for existence check.")

    chunk_masks = []
    print("Checking skeleton existence in cache...")
    with tqdm(total=len(exists_chunks_without_ident), desc="Checking skeleton existence") as pbar:
        for chunk_without_ident in exists_chunks_without_ident: 
            # Query the CAVE client to check if skeletons exist for the given root IDs
            chunk_in_cache = client.skeleton.skeletons_exist(
                root_ids=chunk_without_ident, 
                datastack_name=datastack_name, 
                skeleton_version=skeleton_version
            )
            # Create a boolean mask: True if skeleton exists, False otherwise
            chunk_mask = [True if chunk_in_cache.get(root_id_without_ident) else False for root_id_without_ident in chunk_without_ident]
            chunk_masks.append(chunk_mask)
            pbar.update()
    
    # Flatten the list of masks into a single boolean array
    flat_chunk_masks = sum(chunk_masks, [])
    
    # Convert the list of roots to a NumPy array to enable boolean indexing
    roots_np = np.array(roots)

    # Filter the original roots (with identifiers) using the flattened mask
    roots_with_skeletons = roots_np[flat_chunk_masks].tolist()
    
    print(f"Original roots count: {len(roots)}")
    print(f"Roots with existing skeletons count: {len(roots_with_skeletons)}")
    
    # Save the filtered list of roots (only those for which skeletons exist)
    data_utils.save_txt(post_generate_roots_path, roots_with_skeletons)
    print(f"Saved roots with existing skeletons to: {post_generate_roots_path}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time # Calculate elapsed time
    print(f"\nProcess completed in {elapsed_time:.2f} seconds.")