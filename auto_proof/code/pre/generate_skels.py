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

def main(data_config):
    """
        TODO: Fill in
        TODO: Assuming that roots passed in already includes the proofread root list, add this logic to process raw edits
    """

    data_dir = data_config['data_dir']
    features_dir = f'{data_dir}{data_config['features']['features_dir']}'
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    mat_version_start = data_config['client']['mat_version_start']
    mat_version_end = data_config['client']['mat_version_end']
    roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'

    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/roots_diff_from_curr.txt")
    # roots = data_utils.load_txt(f'{roots_dir}{data_config['proofread']['post_proofread_roots']}')
    # roots = ['864691135155575396_000'] # non proofread
    # roots = ['864691135865167998_000'] # proofread
    # roots = ['864691135155575396_000', '864691135865167998_000']

    # root_to_rep_coord = data_utils.load_pickle_dict(data_config['raw_edits']['root_to_rep'])
    # proofread_roots = data_utils.load_txt(data_config['raw_edits']['proofread_roots'])
    client = data_utils.create_client(data_config)
    datastack_name = data_config['client']['datastack_name']
    skeleton_version = data_config['features']['skeleton_version']

    

    print("roots original len", len(roots))
    files = glob.glob(f'{features_dir}*')
    existing_roots = [files[i][-27:-5] for i in range(len(files))]
    roots = np.setdiff1d(roots, existing_roots)
    print("roots before len", len(roots))

    bulk_chunk_size = data_config['features']['generate_chunk_size']
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
    data_config = data_utils.get_data_config()
    main(data_config)