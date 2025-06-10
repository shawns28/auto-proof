from auto_proof.code.dataset import AutoProofDataset, prune_edges, build_dataloader, edge_list_to_adjency
from auto_proof.code.pre import data_utils
from auto_proof.code.model import create_model
from auto_proof.code.visualize import visualize_whole_cell
from auto_proof.code.pre.skeletonize import get_skel, process_skel, bfs
from auto_proof.code.pre.map_pe import map_pe_wrapper
from auto_proof.code.pre.roots_at import get_roots_at
from auto_proof.code.pre.labels import create_labels
from auto_proof.code.connectomics.reader import EmbeddingReader
from auto_proof.code.connectomics.sharding import md5_shard
from auto_proof.code.pre.segclr import get_segclr_emb, sharder, get_roots_at_seglcr_version
from auto_proof.code.pre.distance import create_dist

from collections import deque
import torch
import torch.nn as nn
import json
import numpy as np
from tqdm import tqdm
from torch.multiprocessing import Manager
import networkx as nx
import h5py
from torch.utils.data import Dataset, DataLoader, random_split
import time
import matplotlib.pyplot as plt
import os
import gcsfs
import networkx as nx
from cloudvolume import CloudVolume
import pyvista as pv

# - get the skeleton for a full root
# - get the features for a full root
# - get the labels for a full root
# - get segclr for the full root

# Now we should have the inputs and outputs for the model
# - get a random 250 node subgraph by picking a random seed node and running bfs for 250
# - run it through the model and mark the 100 core with the outputs
# - mark the 100 core as seen
# - pick another starting point and it can't be in anything in seen and get another 250
# - do this until all are seen, when conflicts arise in the seen, bias error for now


# In production I assume things would change and you would need to care about version of root, labels, segclr
def get_root_dict(root, client_config, data_config, config):
    whole_cell_dir = config['whole_cell']['data_dir']
    root_path = f'{whole_cell_dir}{root}.hdf5'
    if os.path.exists(root_path):
        print("loading root dict for", root)
        data_dict = {}
        try:
            with h5py.File(root_path, 'r') as hf:
                for key in hf.keys():
                    item = hf[key]
                    if isinstance(item, h5py.Dataset):
                        data_dict[key] = item[()]
                    else:
                        print(f"Warning: Item '{key}' is a group and will be skipped as this function only handles base-level datasets.")
        except Exception as e:
            print(f"Error opening or reading HDF5 file: {e}")
            return {}
        return data_dict
        
    else: # create the root
        print("creating root dict for", root)
        # skeletonize
        client, datastack_name, mat_version_start, mat_version_end = data_utils.create_client(client_config)
        
        print("getting skeleton")
        status, e, skel_dict = get_skel(datastack_name, data_config['features']['skeleton_version'], root, client)
        if status == False:
            raise Exception("Failed to get skel for root", root, "eror:", e)
        root_dict = {}
        root_dict['vertices'] = skel_dict['vertices']
        root_dict['edges'] = skel_dict['edges']
        root_dict['radius'] = skel_dict['radius']
        
        # map pe
        status, e, map_pe = map_pe_wrapper(data_config['features']['pos_enc_dim'], root_dict['edges'], len(root_dict['vertices']))
        if status == False:
            raise Exception("Failed map pe for root", root, "eror:", e)
        root_dict['map_pe'] = map_pe

        # roots at for labels
        print("getting roots at for labels")
        latest_version = data_config['labels']['latest_mat_version']
        seg_path = data_config['segmentation'][f'precomputed_{latest_version}']
        cv_seg = CloudVolume(seg_path, use_https=True)
        resolution = np.array(data_config['segmentation']['resolution'])
        status, e, root_at_arr = get_roots_at(root_dict['vertices'], cv_seg, resolution)
        if status == False:
            raise Exception("Failed to get roots at for root", root, "eror:", e)
        root_dict['roots_at'] = root_at_arr

        # labels
        proofread_mat_version1 = data_config['proofread']['mat_versions'][0]
        proofread_mat_version2 = data_config['proofread']['mat_versions'][1]
        proofread_roots_path = f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{proofread_mat_version1}_{proofread_mat_version2}.txt'
        proofread_roots = data_utils.load_txt(proofread_roots_path)
        ignore_edge_ccs = data_config['labels']['ignore_edge_ccs']
        labels, confidences = create_labels(root, root_dict['roots_at'], ignore_edge_ccs, root_dict['edges'], proofread_roots)
        root_dict['labels'] = labels
        root_dict['confidences'] = confidences

        dist = create_dist(root, root_dict['edges'], root_dict['labels'])
        root_dict['dist'] = dist
        
        # root at segclr
        print("getting roots at for segclr")
        mat_versions = data_config['segclr']['mat_versions']
        segmentation_version = get_roots_at_seglcr_version(root, data_config['data_dir'], mat_versions, True)
        if segmentation_version == 1300:
            raise Exception("shouldn't have segmentation version 1300 for segclr")
        seg_path = data_config['segmentation'][f'precomputed_{segmentation_version}']
        cv_seg = CloudVolume(seg_path, use_https=True)
        resolution = np.array(data_config['segmentation']['resolution'])
        
        status, e, root_at_arr = get_roots_at(root_dict['vertices'], cv_seg, resolution)
        if status == False:
            raise Exception("Failed to get roots at for root", root, "eror:", e)
        root_dict['roots_at_segclr'] = root_at_arr

        # segclr emb
        print("getting segclr emb")
        filesystem = gcsfs.GCSFileSystem(token='anon')
        num_shards = data_config['segclr'][f'num_shards_{segmentation_version}']
        bytewidth = data_config['segclr'][f'bytewidth_{segmentation_version}']
        url = data_config['segclr'][f'url_{segmentation_version}']
        embedding_reader = EmbeddingReader(filesystem, url, sharder, num_shards, bytewidth)
        emb_dim = data_config['segclr']['emb_dim']
        visualize_radius = data_config['segclr']['visualize_radius']
        small_radius = data_config['segclr']['small_radius']
        large_radius = data_config['segclr']['large_radius']

        status, e, segclr_emb, has_emb = get_segclr_emb(root, root_dict['vertices'], root_dict['edges'], root_dict['roots_at_segclr'], embedding_reader, emb_dim, visualize_radius, small_radius, large_radius)
        if status == False:
            raise Exception("Failed to segclr for root", root, "eror:", e)
        
        root_dict['segclr_emb'] = segclr_emb
        root_dict['has_emb'] = has_emb

        with h5py.File(root_path, 'a') as root_f:
            for attribute in root_dict:
                root_f.create_dataset(attribute, data=root_dict[attribute])
        return root_dict

def get_whole_cell_output(root_dict, config, model, device):
    num_vertices = len(root_dict['vertices'])
    print("num vertices", num_vertices)
    if num_vertices > 4000:
        raise Exception("number of vertices too big to visualize with mesh") 
    seen_nodes = set()
    output_sum = torch.zeros(num_vertices, dtype=torch.float).to(device)
    output_count = torch.zeros(num_vertices, dtype=torch.float).to(device)
   
    # output = torch.full((10,), -1, dtype=torch.float)
    fov = config['loader']['fov']
    box_cutoff = config['data']['box_cutoff']
    q = deque(list(range(num_vertices)))

    edges = root_dict['edges']

    # create graph
    g = nx.Graph()
    g.add_nodes_from(range(num_vertices))
    g.add_edges_from(edges)
    # features = ['vertices', 'radius', 'map_pe', 'has_emb', 'segclr_emb']
    # labels = ['labels', 'confidences', 'dist']

    # convert to torch from numpy
    # TODO: This is currently only configured for the baseline parameters
    # and in the future should reflect models configs
    vertices = torch.from_numpy(root_dict['vertices'])     
    pos_enc = torch.from_numpy(root_dict['map_pe'])
    radius = torch.from_numpy(root_dict['radius']).unsqueeze(1)
    dist_to_error = torch.from_numpy(root_dict['dist']).unsqueeze(1)
    labels = torch.from_numpy(root_dict['labels']).int().unsqueeze(1)
    confidences = torch.from_numpy(root_dict['confidences']).int().unsqueeze(1)

    relative_vertices = True
    mean_vertices = torch.zeros(3)
    if relative_vertices:
        mean_vertices = torch.mean(vertices, dim=0, keepdim=True) # normalize vertices
        vertices = vertices - mean_vertices

    zscore_radius = True
    radius_mean = torch.from_numpy(np.load("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/radius_mean.npy"))
    radius_std = torch.from_numpy(np.load("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/radius_std.npy"))
    if zscore_radius:
        radius = (radius - radius_mean) / radius_std

    use_segclr = True
    # create input matrix
    if use_segclr:
            segclr = torch.from_numpy(root_dict['segclr_emb']).float()
            has_emb = torch.from_numpy(root_dict['has_emb']).unsqueeze(1)
            input = torch.cat((vertices, radius, pos_enc, segclr, has_emb), dim=1)
    else:
        input = torch.cat((vertices, radius, pos_enc), dim=1)
    
    # TODO: Need to pad the cell if its smaller than fov
    while q:
        seed = q.popleft()
        if seed not in seen_nodes:
            rank = torch.from_numpy(bfs(g, seed, num_vertices))
            
            indices = torch.where(rank < fov)[0]
            sub_input = input[indices].float().to(device).unsqueeze(0)
            sub_labels = labels[indices].float().to(device)
            sub_confidences = confidences[indices].float().to(device)
            sub_dist_to_error = dist_to_error[indices].float().to(device)
            sub_rank = rank[indices].float().to(device)
            sub_edges = prune_edges(edges, indices)
            sub_adj = edge_list_to_adjency(sub_edges, num_vertices, fov).float().to(device).unsqueeze(0)

            sub_output = model(sub_input, sub_adj) # (1, fov, 1)
            sigmoid = nn.Sigmoid()
            sub_output = sigmoid(sub_output) # (1, fov, 1)
            sub_output = sub_output.squeeze(0) # (fov, 1)
            sub_output = sub_output.squeeze(1)            

            box_output_indices = torch.where(rank < box_cutoff)[0]

            ranks_within_model_fov = rank[indices]
            mask_for_core_update = (ranks_within_model_fov < box_cutoff)
            sub_output_for_core_update = sub_output[mask_for_core_update]

            output_sum[box_output_indices] += sub_output_for_core_update
            output_count[box_output_indices] += 1

            seen_nodes.update(box_output_indices.tolist())

    final_output = torch.where(output_count > 0, output_sum / output_count, output_sum)

    return final_output.detach().cpu().numpy()

def get_root_mesh(root, client_config):
    client, _, _, _ = data_utils.create_client(client_config)  
    cv = client.info.segmentation_cloudvolume(progress=False)
    root_without_num = int(root[:-4]) # Removing _000 for mesh retrieval
    mesh = cv.mesh.get(
        root_without_num, deduplicate_chunk_boundaries=False, remove_duplicate_vertices=False
    )[root_without_num]
    
    root_mesh = pv.make_tri_mesh(mesh.vertices, mesh.faces)
    return root_mesh
   
if __name__ == "__main__":
    client_config = data_utils.get_config('client')
    data_config = data_utils.get_config('data')
    config = data_utils.get_config('base')

    # root = "864691134918370314_000" # proofread
    root = "864691135864976604_000" # good example with confident nodes and big and errors
    # root = "864691134928303015_000"
    # root = "864691135785368358_000"
    root = "864691135594727723_000"
    root = "864691135385310293_000"
    root = "864691137196879425_000"
    root_dict = get_root_dict(root, client_config, data_config, config)

    ckpt_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/ckpt/"   
    run_id = 'AUT-330' # new baseline
    epoch = 60
    # run_id = 'AUT-334' # 500 fov
    # epoch = 45
    # run_id = 'AUT-335' # 500 fov 100 weight
    # epoch = 45
    # run_id = 'AUT-330' # ignoring edge errors
    # epoch = 35
    run_dir = f'{ckpt_dir}{run_id}/'
    with open(f'{run_dir}config.json', 'r') as f:
        config = json.load(f)

    model = create_model(config)
    ckpt_path = f'{run_dir}model_{epoch}.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    output = get_whole_cell_output(root_dict, config, model, device)
    print("output", output)

    print("getting mesh")
    mesh = get_root_mesh(root, client_config)

    # create a potentially separate visual method very similar to usual
    print("visualizing")
    save_dir = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/'
    threshold = 0.05
    save_path = f'{save_dir}whole_cell/{root}_250fov_thres005_averageconflict.html' # Change thres accordingly
    visualize_whole_cell(root_dict['vertices'], root_dict['edges'], root_dict['labels'], root_dict['confidences'], output, mesh, threshold, save_path)