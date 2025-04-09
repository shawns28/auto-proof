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
    # num_processes = data_config['multiprocessing']['num_processes']

    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/roots_diff_from_curr.txt")
    # roots = data_utils.load_txt(data_config['features']['pre_root_path'])
    # roots = ['864691135155575396_000'] # non proofread
    # roots = ['864691135865167998_000'] # proofread
    # roots = ['864691135155575396_000', '864691135865167998_000']

    # root_to_rep_coord = data_utils.load_pickle_dict(data_config['features']['root_to_rep'])
    # proofread_roots = data_utils.load_txt(data_config['proofread']['proofread_roots'])
    client = data_utils.create_client(data_config)
    datastack_name = data_config['client']['datastack_name']
    skeleton_version = data_config['features']['skeleton_version']

    features_dir = data_config['features']['features_dir']
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

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

    # exist_chunk_size = data_config['features']['exists_chunk_size']
    # exists_chunks = [roots[i:i + exist_chunk_size] for i in range(0, len(roots), exist_chunk_size)]
    # exists_chunks_without_ident = [[int(root[:-4]) for root in chunk] for chunk in exists_chunks]

    # for chunk_without_ident in exists_chunks_without_ident:
    #     curr_chunk_without_ident = chunk_without_ident
    #     prev_count = -1
    #     curr_count = len(curr_chunk_without_ident)
    #     while curr_count > 0 and curr_count != prev_count:
    #         prev_count = len(curr_chunk_without_ident)
    #         chunk_in_cache = client.skeleton.skeletons_exist(root_ids=curr_chunk_without_ident, datastack_name=datastack_name, skeleton_version=skeleton_version)
    #         chunk_mask = [True if chunk_in_cache[root_id_without_ident] else False for root_id_without_ident in curr_chunk_without_ident]
    #         roots_to_process = curr_chunk_without_ident[chunk_mask]
    #         curr_chunk_without_ident = curr_chunk_without_ident[~chunk_mask]
    #         curr_count = len(curr_chunk_without_ident)

    #         # TODO: Currently wait for Keith to give me a better solution
    #         args_list = []
    #         for root in roots_to_process:
    #             root_without_identifier = int(root[:-4])
    #             if root_without_identifier not in root_to_rep_coord:
    #                 rep = None
    #             else:
    #                 rep = root_to_rep_coord[root_without_identifier]
    #             args_list.append((root, rep, client))

    #         num_processes = data_config['multiprocessing']['num_processes']
    #         with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
    #             for _ in pool.imap_unordered(process_root_features, args_list):
    #                 pbar.update()



    #         if curr_count == 0:
    #             break
    #         time.sleep(300) # 5 min
        

    # chunk_masks = sum(chunk_masks, [])
    # print("len chunk masks", len(chunk_masks))

    # roots = roots[chunk_masks]
    # print("roots len after checking if in cache", len(roots))

    # args_list = []
    # for root in roots:
    #     root_without_identifier = int(root[:-4])
    #     if root_without_identifier not in root_to_rep_coord:
    #         rep = None
    #     else:
    #         rep = root_to_rep_coord[root_without_identifier]
    #     args_list.append((root, rep, client))

    # num_processes = data_config['multiprocessing']['num_processes']
    # with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
    #     for _ in pool.imap_unordered(process_root_features, args_list):
    #         pbar.update()

    # files = glob.glob(f'{data_config['features']['features_dir']}*')
    # roots = [files[i][-27:-5] for i in range(len(files))]
    # data_utils.save_txt(data_config['features']['post_root_path'], roots)

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
    status, e, skel_dict = get_skel(data_config['client']['datastack_name'], data_config['features']['skeleton_version'], root, client)
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