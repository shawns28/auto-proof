from auto_proof.code.pre import data_utils
from auto_proof.code.pre.skeletonize import get_skel, process_skel
from auto_proof.code.pre.map_pe import map_pe_wrapper

import os
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import glob

def main(data_config):
    """
        TODO: Fill in
        TODO: Assuming that roots passed in already includes the proofread root list, add this logic to process raw edits
    """
    data_config, chunk_num, num_chunks, num_processes = data_utils.get_num_chunk_and_processes(data_config)
    # num_processes = data_config['multiprocessing']['num_processes']

    roots = data_utils.load_txt(data_config['features']['pre_root_path'])
    # roots = ['864691135155575396_000'] # non proofread
    # roots = ['864691135865167998_000'] # proofread
    # roots = ['864691135155575396_000', '864691135865167998_000']
    roots = data_utils.get_roots_chunk(roots, chunk_num=chunk_num, num_chunks=num_chunks)

    root_to_rep_coord = data_utils.load_pickle_dict(data_config['features']['root_to_rep'])
    # proofread_roots = data_utils.load_txt(data_config['proofread']['proofread_roots'])
    client = data_utils.create_client(data_config)

    features_dir = data_config['features']['features_dir']
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    args_list = []
    for root in roots:
        root_without_identifier = int(root[:-4])
        if root_without_identifier not in root_to_rep_coord:
            rep = None
        else:
            rep = root_to_rep_coord[root_without_identifier]
        args_list.append((root, rep, client))

    num_processes = data_config['multiprocessing']['num_processes']
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root_features, args_list):
            pbar.update()

    files = glob.glob(f'{data_config['features']['features_dir']}*')
    roots = [files[i][-27:-5] for i in range(len(files))]
    data_utils.save_txt(data_config['features']['post_root_path'], roots)

def process_root_features(data):
    """
       TODO: Fill in
    """
    root, data_config, rep, client = data
    if rep is None:
        is_proofread = True
    else:
        is_proofread = False

    feature_path = f'{data_config['features']['features_dir']}{root}.hdf5'
    # Skip already processed roots
    if os.path.exists(feature_path):
        return

    # Skeletonize
    status, e, skel_dict = get_skel(data_config['client']['datastack_name'], root, client)
    status, e, feature_dict = process_skel(data_config['features']['cutoff'], data_config['features']['box_cutoff'], is_proofread, rep, skel_dict)

    if status == False:
        print("Failed skel for root", root, "eror:", e)
        return
    
    # map pe
    status, e, map_pe = map_pe_wrapper(data_config['features']['pos_enc_dim'], root, feature_dict['edges'])
    if status == False:
        print("Failed map pe for root", root, "eror:", e)
        return
    feature_dict['map_pe'] = map_pe

    with h5py.File(feature_path, 'a') as feat_f:
        for feature in feature_dict:
            feat_f.create_dataset(feature, data=feature_dict[feature])

if __name__ == "__main__":
    data_config = data_utils.get_data_config()
    main(data_config)