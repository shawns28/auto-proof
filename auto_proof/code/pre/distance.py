from auto_proof.code.pre import data_utils

import graph_tool.all as gt
import numpy as np
from tqdm import tqdm
import multiprocessing
import glob
import os
import h5py
import argparse

# Creates distances to closest labeled error for each node

def create_dist(config, dist_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chunk_num", help="chunk num")
    parser.add_argument("-n", "--num_workers", help="num workers")
    args = parser.parse_args()
    if args.num_workers:
        config['loader']['num_workers'] = int(args.num_workers)
    chunk_num = 1
    num_chunks = config['data']['num_chunks']
    if args.chunk_num:
        chunk_num = int(args.chunk_num)
    else: # No chunking
        num_chunks = 1

    roots = data_utils.load_txt(config['data']['root_path'])
    # roots = [864691135937424949]
    # roots = [864691135778235581]
    roots = data_utils.get_roots_chunk(config, roots, chunk_num=chunk_num, num_chunks=num_chunks)

    num_processes = config['loader']['num_workers']
    features_directory = config['data']['features_dir']
    args_list = list([(root, features_directory, dist_dir) for root in roots])

    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for _ in pool.imap_unordered(process_root, args_list):
            pbar.update()

def process_root(args):
    try:
        root, features_directory, dist_dir = args
        dist_path = f'{dist_dir}dist_{root}.hdf5'

        # Skip already processed roots
        if os.path.exists(dist_path):
            return

        root_feat_path = f'{features_directory}{root}.hdf5'
        with h5py.File(root_feat_path, 'r') as f:
            edges = f['edges'][:]
            g = gt.Graph(edges, directed=False)
            labels = f['label'][:]
            distances = get_distances(g, labels)

            with h5py.File(dist_path, 'a') as dist_f:
                dist_f.create_dataset('dist', data=distances)

    except Exception as e:
        print(e)
        return

def get_distances(g, labels):
    zero_vertices = [v for v in g.vertices() if labels[int(v)] == 0]
    distances = g.new_vertex_property("int")
    distances.a[:] = np.iinfo(np.int32).max

    if not zero_vertices:
        return distances.a
    
    for v in zero_vertices:
        shortest_dist = gt.shortest_distance(g, source=v)
        for u in g.vertices():
            if shortest_dist[u] < distances[u]:
                distances[u] = shortest_dist[u]
    return distances.a

def save_dist_roots(dist_dir, post_dist_roots_file):
    files = glob.glob(f'{dist_dir}*')
    roots = [files[i][-23:-5] for i in range(len(files))]
    print(roots[0])
    data_utils.save_txt(post_dist_roots_file, roots)

if __name__ == "__main__":
    config = data_utils.get_config()
    config['data']['features_dir'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/features/"
    config['data']['root_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/proofread_roots_in_train_roots_369502_913_conv.txt"
    dist_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/dist/"
    create_dist(config, dist_dir)

    post_dist_roots_file = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/post_dist_roots.txt"
    save_dist_roots(dist_dir, post_dist_roots_file)