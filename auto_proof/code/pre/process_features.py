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
    data_config, chunk_num, num_chunks, num_processes = data_utils.get_num_chunk_and_processes(data_config)

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
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/roots_diff_from_curr.txt")
    roots = data_utils.load_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{data_config['proofread']['post_proofread_roots']}')
    # roots = ['864691135155575396_000'] # non proofread
    # roots = ['864691135865167998_000'] # proofread
    # roots = ['864691135155575396_000', '864691135865167998_000']

    root_to_rep_coord = data_utils.load_pickle_dict(f'{dicts_dir}{data_config['raw_edits']['root_to_rep']}')
    # proofread_roots = data_utils.load_txt(data_config['raw_edits']['proofread_roots'])

    skeleton_version = data_config['features']['skeleton_version']

    # TODO: Change after using
    post_generate_roots_path = f'{roots_dir}{data_config['features']['post_generate_roots']}'
    # post_generate_roots_path = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_unique_copied.txt"
    if not os.path.exists(post_generate_roots_path):
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
    else:
        roots = data_utils.load_txt(post_generate_roots_path)
    print("roots after checking exists in cache len", len(roots))
    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)

    # roots = ['864691135778235581_000', '864691131576191498_000', '864691131614916610_000', '864691131615287139_000','864691133239736250_000', '864691133239750609_000', '864691133248462820_000']

    args_list = []
    for root in roots:
        root_without_identifier = int(root[:-4])
        if root_without_identifier not in root_to_rep_coord:
            rep = None
        else:
            print("root has a coord", root)
            rep = root_to_rep_coord[root_without_identifier]
        args_list.append((root, data_config, rep, client, datastack_name))

    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root_features, args_list):
            pbar.update()

    files = glob.glob(f'{features_dir}*')
    roots = [files[i][-27:-5] for i in range(len(files))]
    data_utils.save_txt(f'{roots_dir}{data_config['features']['post_feature_roots']}', roots)

def process_root_features(data):
    """
       TODO: Fill in
    """
    root, data_config, rep, client, datastack_name = data
    if rep is None:
        is_proofread = True
    else:
        is_proofread = False

    feature_path = f'{data_config['data_dir']}{data_config['features']['features_dir']}{root}.hdf5'
    # Skip already processed roots
    if os.path.exists(feature_path):
        return

    # Skeletonize
    status, e, skel_dict = get_skel(datastack_name, data_config['features']['skeleton_version'], root, client)
    if status == False:
        print("Failed to get skel for root", root, "eror:", e)
        return
    
    status, e, feature_dict = process_skel(data_config['features']['cutoff'], data_config['features']['box_cutoff'], is_proofread, rep, skel_dict)
    if status == False:
        print("Failed to process for root", root, "eror:", e)
        return
    
    # map pe
    status, e, map_pe = map_pe_wrapper(data_config['features']['pos_enc_dim'], feature_dict['edges'], len(feature_dict['vertices']))
    if status == False:
        print("Failed map pe for root", root, "eror:", e)
        return
    feature_dict['map_pe'] = map_pe

    # print(feature_dict)

    with h5py.File(feature_path, 'a') as feat_f:
        for feature in feature_dict:
            feat_f.create_dataset(feature, data=feature_dict[feature])

if __name__ == "__main__":
    main()