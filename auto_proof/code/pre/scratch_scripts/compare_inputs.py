from auto_proof.code.pre import data_utils
from auto_proof.code.dataset import edge_list_to_adjency, prune_edges

import os
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import torch

def convert(data):
    root, new_feature_dir, new_labels_dir, new_roots_at_dir, og_feature_dir, og_map_dir, og_rank_dir, og_dist_dir = data
    # print("root", root)
    new_feature_path = f'{new_feature_dir}{root}.hdf5'
    new_labels_path = f'{new_labels_dir}{root}.hdf5'
    new_roots_at_path = f'{new_roots_at_dir}{root}.hdf5'
    og_feature_path = f'{og_feature_dir}{root}.hdf5'
    og_map_path = f'{og_map_dir}map_{root}.hdf5'
    og_rank_path = f'{og_rank_dir}rank_{root}.hdf5'
    og_dist_path = f'{og_dist_dir}dist_{root}.hdf5'

    with h5py.File(new_roots_at_path, 'r') as new_roots_f, h5py.File(new_feature_path, 'r') as new_feat_f, h5py.File(new_labels_path, 'r') as new_labels_f, h5py.File(og_feature_path, 'r') as og_feat_f, h5py.File(og_map_path, 'r') as og_map_f, h5py.File(og_rank_path, 'r') as og_rank_f, h5py.File(og_dist_path, 'r') as og_dist_f :
        fov = 250
        # og_vertices = torch.from_numpy(og_feat_f['vertices'][:])
        # og_edges = og_feat_f['edges'][:]
        og_labels = torch.from_numpy(og_feat_f['label'][:]).unsqueeze(1).int()
        og_confidences = torch.from_numpy(og_feat_f['confidence'][:]).unsqueeze(1).int()
        # og_radius = torch.from_numpy(og_feat_f['radius'][:]).unsqueeze(1)
        og_rank = torch.from_numpy(og_rank_f['rank_100'][:]).unsqueeze(1)
        # og_pos_enc = torch.from_numpy(og_map_f['map_pe'][:])
        # og_dist = torch.from_numpy(og_dist_f['dist'][:]).unsqueeze(1)
        # og_roots_at = torch.from_numpy(og_feat_f['root_943'][:]).unsqueeze(1)
        # # print(og_roots_at)
        # og_input = torch.cat((og_vertices, og_radius, og_pos_enc), dim=1)

        size = len(og_labels) 
        if size > fov:
            indices = torch.where(og_rank < fov)[0]
            # og_input = og_input[indices]
            og_labels = og_labels[indices]
            og_confidences = og_confidences[indices]
            # og_dist = og_dist[indices]
            # og_rank = og_rank[indices]
            # og_edges = prune_edges(og_edges, indices)
        # elif(size < fov):
        #     og_input = torch.cat((og_input, torch.zeros((fov - size, og_input.shape[1]))), dim=0)
        #     og_labels = torch.cat((og_labels, torch.full((fov - size, 1), -1)))
        #     og_confidences = torch.cat((og_confidences, torch.full((fov - size, 1), -1)))
        #     og_dist = torch.cat((og_dist, torch.full((fov - size, 1), -1)))
        #     og_rank = torch.cat((og_rank, torch.full((fov - size, 1), -1)))

        
        # og_adj = edge_list_to_adjency(og_edges, size, fov)
        # torch.set_printoptions(threshold=100000)
        # print("og input", og_input)
        # print("og_labels", og_labels)
        # print("og_confidences", og_confidences)
        # print("og_dist", og_dist)
        # print("og_rank", og_rank)
        # print("og_edges", og_edges)
        # print("og adj", og_adj)
    
        # vertices = torch.from_numpy(new_feat_f['vertices'][:])
        # pos_enc = torch.from_numpy(new_feat_f['map_pe'][:])

        rank = torch.from_numpy(new_feat_f[f'rank'][:]).unsqueeze(1)
        # radius = torch.from_numpy(new_feat_f['radius'][:]).unsqueeze(1)
        # edges = new_feat_f['edges'][:]

        # dist_to_error = torch.from_numpy(new_labels_f['dist'][:]).unsqueeze(1)
        labels = torch.from_numpy(new_labels_f['labels'][:]).int().unsqueeze(1)
        confidences = torch.from_numpy(new_labels_f['confidences'][:]).int().unsqueeze(1)
        
        # new_roots_at = torch.from_numpy(new_roots_f['roots_at'][:]).unsqueeze(1)
        # # print(new_roots_at)
        # input = torch.cat((vertices, radius, pos_enc), dim=1)
        size = len(labels)
        if size > fov:
            indices = torch.where(rank < fov)[0]
            # input = input[indices]
            labels = labels[indices]
            confidences = confidences[indices]
            # dist_to_error = dist_to_error[indices]
            # rank = rank[indices]
            # edges = prune_edges(edges, indices)
        # elif(size < fov):
        #     input = torch.cat((input, torch.zeros((fov - size, input.shape[1]))), dim=0)
        #     labels = torch.cat((labels, torch.full((fov - size, 1), -1)))
        #     confidences = torch.cat((confidences, torch.full((fov - size, 1), -1)))
        #     dist_to_error = torch.cat((dist_to_error, torch.full((fov - size, 1), -1)))
        #     rank = torch.cat((rank, torch.full((fov - size, 1), -1)))

        # adj = edge_list_to_adjency(edges, size, fov)
        # torch.set_printoptions(threshold=100000)
        # print("input", input)
        # print("labels", labels)
        # print("confidences", confidences)
        # print("dist", dist_to_error)
        # print("rank", rank)
        # print("edges", edges)
        # print("adj", adj)
        try: 
            # assert torch.equal(og_input, input)
            assert len(og_labels) == len(labels)
            
            og_labels_mask = og_labels != -1
            labels_mask = labels != -1
            assert torch.equal(og_labels_mask, labels_mask)
            og_labels = og_labels[og_labels_mask]
            labels = labels[labels_mask]
            og_confidences = og_confidences[og_labels_mask]
            confidences = confidences[labels_mask]
            labels = 1 - labels
            assert torch.equal(og_labels, labels)
            assert torch.equal(og_confidences, confidences)
            # assert torch.equal(og_rank, rank)
            # assert torch.equal(og_dist, dist_to_error)
            # assert np.array_equal(og_edges, edges)
            # assert torch.equal(og_adj, adj)
        except Exception as e:
            # print("root", root, "error", e)
            # print("og_confidences", og_confidences)
            # print("   confidences", confidences)
            return root
        # print("all match")
        return ''

# roots = ['864691133772909480_000','864691136334777523_000']
# roots = ['864691133659391257_000']
roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_359227/all_roots.txt")
print(len(roots))
to_remove = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/proofread/943.txt")
to_remove = [str(root) + '_000' for root in to_remove]
roots = np.setdiff1d(roots, to_remove)
print(len(roots))

# roots = ['864691134295024204_000']

og_feature_dir =  "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features_conf/"
og_rank_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ranks/"
og_map_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/map_pes/"
og_dist_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/dist/"

new_feature_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/features/"
new_labels_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/labels_at_1300_wrong/"
new_roots_at_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_at_1300/"

conf_change_roots = []
num_processes = 64
args_list = [(root, new_feature_dir, new_labels_dir, new_roots_at_dir, og_feature_dir, og_map_dir, og_rank_dir, og_dist_dir) for root in roots]
with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(args_list)) as pbar:
    for root in pool.imap_unordered(convert, args_list):
        if root != '':
            conf_change_roots.append(root)
        pbar.update()

print(len(conf_change_roots))
data_utils.save_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_359227/conf_change_roots.txt", conf_change_roots)