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

    root_to_rep_coord = data_utils.load_pickle_dict(f'{dicts_dir}{data_config['raw_edits']['root_to_rep']}')
    # proofread_roots = data_utils.load_txt(data_config['raw_edits']['proofread_roots'])

    skeleton_version = data_config['features']['skeleton_version']

    # TODO: Move this code to a separate file since it should only be ran once and then this file should always just load in post generate roots so that the chunks match up correctly
    post_generate_roots_path = f'{roots_dir}{data_config['features']['post_generate_roots']}'
    # roots = data_utils.load_txt(post_generate_roots_path)
    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/roots_diff_from_curr_2.txt")
    # roots = ['864691136619433869_002']
    # roots = ['864691136926085706_000']
    print("roots len", len(roots))
    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)

    # roots = ['864691135778235581_000', '864691131576191498_000', '864691131614916610_000', '864691131615287139_000','864691133239736250_000', '864691133239750609_000', '864691133248462820_000']

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
        # TODO: If it fails here create an empty file with the root name without identifier for keith
        root_without_iden = root[:-4]
        error_file_path = f"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/error_roots/{root_without_iden}"
        with open(error_file_path, 'w') as file:
            pass
        return
    
    status, e, feature_dict = process_skel(data_config['features']['box_cutoff'], data_config['features']['cutoff'], is_proofread, rep, skel_dict)
    if len(feature_dict['vertices'] == 1):
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
    main()