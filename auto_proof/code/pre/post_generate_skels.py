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
    """
        TODO: Fill in
        TODO: Assuming that roots passed in already includes the proofread root list, add this logic to process raw edits
    """
    data_config = data_utils.get_config('data')
    client_config = data_utils.get_config('client')

    data_dir = data_config['data_dir']
    client, datastack_name, mat_version_start, mat_version_end = data_utils.create_client(client_config)

    roots_dir = f'{data_dir}roots_{mat_version_start}_{mat_version_end}/'
    dicts_dir = f'{data_dir}dicts_{mat_version_start}_{mat_version_end}/'
    features_dir = f'{data_dir}{data_config['features']['features_dir']}'
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # TODO: Change after using   
    #TODO: Should be able to load in all of the roots due to the check below and also checking if a root exists in the cache
    # roots = data_utils.load_txt(f'{roots_dir}{data_config['proofread']['post_proofread_roots']}')
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_unique_copied.txt")
    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/roots_diff_from_curr.txt")
    # roots = data_utils.load_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{data_config['proofread']['post_proofread_roots']}')
    # roots = ['864691135155575396_000'] # non proofread
    # roots = ['864691135865167998_000'] # proofread
    # roots = ['864691135155575396_000', '864691135865167998_000']

    # proofread_roots = data_utils.load_txt(data_config['raw_edits']['proofread_roots'])

    # TODO: Move this code to a separate file since it should only be ran once and then this file should always just load in post generate roots so that the chunks match up correctly
    post_generate_roots_path = f'{roots_dir}{data_config['features']['post_generate_roots']}'
    # post_generate_roots_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_unique_copied.txt"
    
    print("roots original len", len(roots))
    files = glob.glob(f'{features_dir}*')
    already_featurized_roots = [files[i][-27:-5] for i in range(len(files))]
    roots = np.setdiff1d(roots, already_featurized_roots)
    print("roots after removing already featurized len", len(roots))

    exist_chunk_size = data_config['features']['exists_chunk_size']
    exists_chunks = [roots[i:i + exist_chunk_size] for i in range(0, len(roots), exist_chunk_size)]
    exists_chunks_without_ident = [[int(root[:-4]) for root in chunk] for chunk in exists_chunks]
    # TODO: Save this as post_generate_roots
    chunk_masks = []
    with tqdm(total=len(exists_chunks_without_ident)) as pbar:
        for chunk_without_ident in exists_chunks_without_ident: 
            chunk_in_cache = client.skeleton.skeletons_exist(root_ids=chunk_without_ident, datastack_name=datastack_name, skeleton_version=skeleton_version)
            chunk_mask = [True if chunk_in_cache[root_id_without_ident] else False for root_id_without_ident in chunk_without_ident]
            chunk_masks.append(chunk_mask)
            pbar.update()
    chunk_masks = sum(chunk_masks, [])
    print(chunk_masks[0])
    print("chunk flattened len", len(chunk_masks))
    print("roots len", len(roots))
    roots = roots[chunk_masks]
    data_utils.save_txt(f'{roots_dir}{data_config['features']['post_generate_roots']}', roots)

if __name__ == "__main__":
    main()