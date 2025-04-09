from auto_proof.code.pre import data_utils
from auto_proof.code.pre.skeletonize import prune_edges

import os
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing

def convert(data):
    root, new_feature_dir, og_feature_dir, og_map_dir, og_rank_dir = data
    new_feature_path = f'{new_feature_dir}{root}.hdf5'
    og_feature_path = f'{og_feature_dir}{root}.hdf5'
    og_map_path = f'{og_map_dir}map_{root}.hdf5'
    og_rank_path = f'{og_rank_dir}rank_{root}.hdf5'

    if os.path.exists(new_feature_path):
        return

    with h5py.File(og_feature_path, 'r') as og_feat_f, h5py.File(og_map_path, 'r') as og_map_f, h5py.File(og_rank_path, 'r') as og_rank_f:
        num_initial_vertices = og_feat_f['num_initial_vertices'][()]
        vertices = og_feat_f['vertices'][:]
        edges = og_feat_f['edges'][:]
        radius = og_feat_f['radius'][:]
        rank = og_rank_f['rank_100'][:]
        map_pe = og_map_f['map_pe'][:]

        mask = rank < 500
        mask = np.where(mask == True)[0]

        rank = rank[mask]
        vertices = vertices[mask]
        radius = radius[mask]
        map_pe = map_pe[mask]

        edges = prune_edges(mask, edges)

        with h5py.File(new_feature_path, 'a') as new_feat_f:
            new_feat_f.create_dataset('vertices', data=vertices)
            new_feat_f.create_dataset('radius', data=radius)
            new_feat_f.create_dataset('rank', data=rank)
            new_feat_f.create_dataset('map_pe', data=map_pe)            
            new_feat_f.create_dataset('edges', data=edges)
            new_feat_f.create_dataset('num_initial_vertices', data=num_initial_vertices)


roots = data_utils.load_txt()

new_feature_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/features/"
og_feature_dir =  "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features_conf/"
og_rank_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ranks/"
og_map_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/map_pes/"

num_processes = 64
args_list = [(root, new_feature_dir, og_feature_dir, og_map_dir, og_rank_dir) for root in result]
with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(args_list)) as pbar:
    for _ in pool.imap_unordered(convert, args_list):
        pbar.update()


