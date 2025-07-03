from auto_proof.code.pre import data_utils

import os
import numpy as np
import glob
import time

def main():
    """Generates skeletons for a list of root IDs using bulk operations.

    This function orchestrates the process of fetching and generating skeletons
    from the CAVE client. It assumes that the input 'post_proofread_roots' file
    already contains the combined list of initial raw edit roots and proofread roots
    with their unique identifiers.

    Steps involved:
    1. Loads the pre-processed list of root IDs (including proofread roots and copies).
    2. Divides the root IDs into chunks compliant with the api for bulk skeleton generation.
    3. Initiates asynchronous bulk skeleton generation requests to the CAVE client
       and tracks the time taken for these operations.

    Args:
        data_config: A dictionary containing data configuration settings,
            typically loaded from a YAML file.
    """
    data_config = data_utils.get_config('data')
    client_config = data_utils.get_config('client')

    data_dir = data_config['data_dir']

    client, datastack_name, mat_version_start, mat_version_end = data_utils.create_client(client_config)
    roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'

    roots = data_utils.load_txt(f'{roots_dir}{data_config['proofread']['post_proofread_roots']}')

    skeleton_version = data_config['features']['skeleton_version']

    bulk_chunk_size = data_config['generate_skels']['generate_chunk_size']
    bulk_chunks = [roots[i:i + bulk_chunk_size] for i in range(0, len(roots), bulk_chunk_size)]
    bulk_chunks_without_ident = [[int(root[:-4]) for root in chunk] for chunk in bulk_chunks]
    print("shape of bulk chunk without ind", len(bulk_chunks_without_ident), len(bulk_chunks_without_ident[0]))

    time_total = 0
    estimated_time_total = 0
    for i, bulk_chunk_without_ident in enumerate(bulk_chunks_without_ident):
        start_time = time.time()
        estimated_time = client.skeleton.generate_bulk_skeletons_async(root_ids=bulk_chunk_without_ident, datastack_name=datastack_name, skeleton_version=skeleton_version)
        end_time = time.time()
        time_taken = end_time - start_time
        print("time for generate at ", i, "is", time_taken)
        time_total += time_taken
        estimated_time_total += estimated_time
    print("estimated time", estimated_time_total)
    print("time total for generate", time_total)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time # Calculate elapsed time
    print(f"\nProcess completed in {elapsed_time:.2f} seconds.")