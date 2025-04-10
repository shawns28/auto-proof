from auto_proof.code.pre import data_utils
# from auto_proof.code.pre.skeletonize import get_skel, process_skel
# from auto_proof.code.pre.map_pe import map_pe_wrapper

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
    data_config, chunk_num, num_chunks, num_processes = data_utils.get_num_chunk_and_processes(data_config)

    data_dir = data_config['data_dir']
    mat_version_start = data_config['client']['mat_version_start']
    mat_version_end = data_config['client']['mat_version_end']
    roots_dir = f'{data_config['data_dir']}roots_{mat_version_start}_{mat_version_end}/'
    dicts_dir = f'{data_dir}dicts_{mat_version_start}_{mat_version_end}/'
    features_dir = f'{data_dir}{data_config['features']['features_dir']}'
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    
    #TODO: Should be able to load in all of the roots due to the check below and also checking if a root exists in the cache
    roots = data_utils.load_txt(f'{roots_dir}{data_config['proofread']['post_proofread_roots']}')
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/roots_diff_from_curr.txt")
    # roots = data_utils.load_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{data_config['proofread']['post_proofread_roots']}')
    # roots = ['864691135155575396_000'] # non proofread
    # roots = ['864691135865167998_000'] # proofread
    # roots = ['864691135155575396_000', '864691135865167998_000']

    root_to_rep_coord = data_utils.load_pickle_dict(f'{dicts_dir}{data_config['raw_edits']['root_to_rep']}')
    # proofread_roots = data_utils.load_txt(data_config['raw_edits']['proofread_roots'])
    client = data_utils.create_client(data_config)
    datastack_name = data_config['client']['datastack_name']
    skeleton_version = data_config['features']['skeleton_version']

    print("roots original len", len(roots))
    files = glob.glob(f'{features_dir}*')
    already_featurized_roots = [files[i][-27:-5] for i in range(len(files))]
    roots = np.setdiff1d(roots, already_featurized_roots)
    print("roots after removing already featurized len", len(roots))

    # TODO: Should just check if all the roots exist in the beginning and then I don't have to check later
    # TODO: Will restructure later since this is easier for now and they haven't designed a better way
    post_generate_roots_path = f'{roots_dir}{data_config['features']['post_generate_roots']}'
    if not os.path.exists(post_generate_roots_path):
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
    print("roots after checking exists in cache", len(roots))

    # TODO: Currently wait for Keith to give me a better solution
    args_list = []
    for root in roots:
        root_without_identifier = int(root[:-4])
        if root_without_identifier not in root_to_rep_coord:
            rep = None
        else:
            rep = root_to_rep_coord[root_without_identifier]
        args_list.append((root, rep, client))

    # num_processes = data_config['multiprocessing']['num_processes']
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root_features, args_list):
            pbar.update()

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

# def process_root_features(data):
#     """
#        TODO: Fill in
#     """
#     root, data_config, rep, client = data
#     if rep is None:
#         is_proofread = True
#     else:
#         is_proofread = False

#     feature_path = f'{data_config['features']['features_dir']}{root}.hdf5'
#     # Skip already processed roots
#     if os.path.exists(feature_path):
#         return

#     # Skeletonize
#     status, e, skel_dict = get_skel(data_config['client']['datastack_name'], data_config['features']['skeleton_version'], root, client)
#     status, e, feature_dict = process_skel(data_config['features']['cutoff'], data_config['features']['box_cutoff'], is_proofread, rep, skel_dict)

#     if status == False:
#         print("Failed skel for root", root, "eror:", e)
#         return
    
#     # map pe
#     status, e, map_pe = map_pe_wrapper(data_config['features']['pos_enc_dim'], root, feature_dict['edges'])
#     if status == False:
#         print("Failed map pe for root", root, "eror:", e)
#         return
#     feature_dict['map_pe'] = map_pe

#     with h5py.File(feature_path, 'a') as feat_f:
#         for feature in feature_dict:
#             feat_f.create_dataset(feature, data=feature_dict[feature])

if __name__ == "__main__":
    data_config = data_utils.get_data_config()
    main(data_config)