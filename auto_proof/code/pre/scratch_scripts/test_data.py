from auto_proof.code.pre import data_utils
from auto_proof.code.dataset import edge_list_to_adjency, prune_edges, adjency_to_edge_list_torch_skip_diag, adjency_to_edge_list_numpy_skip_diag

import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import torch
import time
import glob

def test():
    data_config = data_utils.get_config('data')
    data_dir = data_config['data_dir']
    roots_dir = f'{data_config['data_dir']}roots_{343}_{1300}/'
    latest_version = data_config['labels']['latest_mat_version']
    labels_at_latest_dir = f'{data_dir}{data_config['labels']['labels_at_latest_dir']}{latest_version}/'
    roots_at_latest_dir = f'{data_dir}{data_config['labels']['roots_at_latest_dir']}{latest_version}/'

    # roots = data_utils.load_txt(f'{roots_dir}{data_config['features']['post_feature_roots']}')
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_503534/all_roots.txt")
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_503534/val_conf_no_error_in_box_roots.txt")
    # roots = ['864691135424062237_000']
    # proofread_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943_1300_copied.txt")
    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/train_roots.txt")
    # roots = roots[:10]
    # args_list = list([(root, features_dir, map_pe_dir, seed_index, fov, max_cloud) for root in roots])
    total_count = 0
    total_mean_radius = 0
    total_sum_squares_radius = 0
    total_mean_pos_enc = np.zeros(32)
    total_sum_squares_pos_enc = np.zeros(32)
    total_mean_segclr = np.zeros(64)
    total_sum_squares_segclr = np.zeros(64)
    args_list = list([(root, data_config) for root in roots])
    num_processes = 64
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for count, mean_radius, sum_squares_radius, mean_pos_enc, sum_squares_pos_enc, mean_segclr, sum_squares_segclr in pool.imap_unordered(process_root, args_list):
            total_count += count
            total_mean_radius += mean_radius
            total_sum_squares_radius += sum_squares_radius
            total_mean_pos_enc += mean_pos_enc
            total_sum_squares_pos_enc += sum_squares_pos_enc
            total_mean_segclr += mean_segclr
            total_sum_squares_segclr += sum_squares_segclr
            pbar.update()

    
    radius_mean = total_mean_radius / total_count
    radius_variance = (total_sum_squares_radius / total_count) - (radius_mean ** 2)
    radius_std = np.sqrt(radius_variance)
    print("final radius and std", radius_mean, radius_variance, radius_std)
    np.save("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/radius_mean.npy", radius_mean)
    np.save("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/radius_std.npy", radius_std)
    
    pos_mean = total_mean_pos_enc / total_count
    pos_variance = (total_sum_squares_pos_enc / total_count) - (pos_mean ** 2)
    pos_std = np.sqrt(pos_variance)
    print("final pos enc mean and std", pos_mean, pos_variance, pos_std)
    np.save("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/pos_mean.npy", pos_mean)
    np.save("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/pos_std.npy", pos_variance)
    
    segclr_mean = total_mean_segclr / total_count
    segclr_variance = (total_sum_squares_segclr / total_count) - (segclr_mean ** 2)
    segclr_std = np.sqrt(segclr_variance)
    print("final segclr mean and std", segclr_mean, segclr_variance, segclr_std)
    np.save("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/segclr_mean.npy", segclr_mean)
    np.save("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/segclr_std.npy", segclr_std)

    # print("one len roots", one_len_roots)
    # data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/one_len_roots.txt", one_len_roots)

    # print("total counts", len(roots))

    # result = np.setdiff1d(no_error_roots, proofread_roots)

    # print("total", len(result))
    # print("total - proofread roots len", len(result) - len(proofread_roots))


def process_root(args):
    root, data_config = args

    feature_path = f'{data_config['data_dir']}{data_config['features']['features_dir']}{root}.hdf5'
    # latest_version = data_config['labels']['latest_mat_version']
    # labels_path = f'{data_config['data_dir']}{data_config['labels']['labels_at_latest_dir']}{latest_version}/{root}.hdf5'
    segclr_path = f'{data_config['data_dir']}{data_config['segclr']['segclr_dir']}{root}.hdf5'
    # og_feature_path = f"/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features_conf/{root}.hdf5"
    try:
        # with  h5py.File(feature_path, 'r') as feat_f:
        # with h5py.File(og_feature_path, 'r') as og_feat_f, h5py.File(feature_path, 'r') as feat_f,  h5py.File(labels_path, 'r') as labels_f, h5py.File(segclr_path, 'r') as segclr_f:
        with h5py.File(feature_path, 'r') as feat_f, h5py.File(segclr_path, 'r') as segclr_f:
            # og_v = og_feat_f['vertices'][:]
            # og_e = og_feat_f['edges'][:]
            # print("og v", og_v)
            # print("og e", og_e)
            # vertices = feat_f['vertices'][:]
            radius = feat_f['radius'][:]
            # labels = labels_f['labels'][:]
            # confidences = labels_f['confidences'][:]
            pos_enc = feat_f['map_pe'][:]

            # rank = feat_f[f'rank'][:]

            # edges = feat_f['edges'][:]

            # dist_to_error = labels_f['dist'][:]
                

            segclr = segclr_f['segclr'][:]
            # has_emb = segclr_f['has_emb'][:]
            # # labels = labels.unsqueeze(1).int()
            # # confidence = confidence.unsqueeze(1).int()
            # size = len(labels)
            # box_cutoff = 100
            # rank = feat_f['rank'][:]
            # print(vertices, radius, labels,confidences, pos_enc, rank, edges, dist_to_error, segclr, has_emb)
            # if size > box_cutoff:
            #     indices = np.where(rank < box_cutoff)[0]
            #     labels = labels[indices]

            # if np.any(labels > 0):
            #     return ''
            # else:
            #     return root
            mean_radius = np.sum(radius)
            sum_squares_radius = np.sum(radius ** 2)
            mean_pos_enc = np.sum(pos_enc, axis=0)
            sum_squares_pos_enc = np.sum(pos_enc ** 2, axis=0)
            mean_segclr = np.sum(segclr, axis=0)
            sum_squares_segclr = np.sum(segclr ** 2, axis=0)
            count = len(radius)
            return count, mean_radius, sum_squares_radius, mean_pos_enc, sum_squares_pos_enc, mean_segclr, sum_squares_segclr

            # if size > fov:
            #     indices = torch.where(rank < fov)[0]
            #     # input = input[indices]
            #     labels = labels[indices]
            #     confidence = confidence[indices]
                # edges = prune_edges(edges, indices)
            # elif(size < fov):
            #     input = torch.cat((input, torch.zeros((fov - size, input.shape[1]))), dim=0)
            #     labels = torch.cat((labels, torch.full((fov - size, 1), -1)))
            #     confidence = torch.cat((confidence, torch.full((fov - size, 1), -1)))

            # error_count = len(np.where(labels == 0)[0])
            # conf_count = len(np.where(confidence == 1)[0])
            # match_count = 0
            # error_match_count = 0
            # if error_count == conf_count:
            #     match_count = 1
            #     if error_count != 0:
            #         error_match_count = 1
            # conf_smaller_than_cloud = 0
            # if max_cloud - conf_count > 0:
            #     conf_smaller_than_cloud = 1
            # node_count = min(fov, size)

            # if abs(error_count - conf_count) == 1:
            #     print("c", conf_count, "e", error_count, "n", node_count, "root", root)

            # print("size", size)
            # time_1 = time.time()
            # adj = edge_list_to_adjency(edges, size, fov)
            # time_2 = time.time()
            # print("A normal", adj)
            # # print("A normal Time", time_2 - time_1)


            # time_5 = time.time()
            # edge_list = adjency_to_edge_list_torch_skip_diag(adj)
            # time_6 = time.time()
            # print("torch edges", edge_list)
            # print("torch edge time", time_6 - time_5)

            # time_7 = time.time()
            # edge_list = adjency_to_edge_list_numpy_skip_diag(adj)
            # time_8 = time.time()
            # print("numpy edges", edge_list)
            # print("numpy edge time", time_8 - time_7)

            # return node_count, error_count, conf_count, match_count, error_match_count, conf_smaller_than_cloud
    except Exception as e:
        print("root: ", root, "error: ", e)
        # return root
        return
        # return 0, 0, 0, 0, 0, 0
                
if __name__ == "__main__":

    test()
    