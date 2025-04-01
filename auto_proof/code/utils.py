from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list_torch_skip_diag
from auto_proof.code.pre import data_utils

import torch
import torch.nn as nn
import json
import pyvista as pv
import numpy as np
from tqdm import tqdm
# from cloudvolume import CloudVolume

def get_root_output(model, device, data, root):
    model.eval() # Marking this here due to async
    with torch.no_grad(): # Marking this here due to async
        # For now it's just pulling a specific sample, later this will pull a specific sample in train/val/test
        # model.to(device)
        idx = data.get_root_index(root)
        sample = data.__getitem__(idx)
        _ , input, labels, confidence, dist_to_error, _ , adj = sample 
        input = input.float().to(device).unsqueeze(0) # (1, fov, d)
        labels = labels.float().to(device) # (fov, 1)
        confidence = confidence.float().to(device) # (fov, 1)
        adj = adj.float().to(device).unsqueeze(0) # (1, fov, fov)
        dist_to_error = dist_to_error.float().to(device) # (fov, 1)

        output = model(input, adj) # (1, fov, 1)
        sigmoid = nn.Sigmoid()
        output = sigmoid(output) # (1, fov, 1)
        output = output.squeeze(0) # (fov, 1)

        # original represents the original points that weren't buffered
        mask = labels != -1
        mask = mask.squeeze(-1) # (original)
        
        # Apply mask to get original points
        output = output[mask] # (original, 1)
        input = input.squeeze(0)[mask] # (original, d)
        labels = labels[mask] # (original, 1)
        confidence = confidence[mask] # (original, 1)
        adj = adj.squeeze(0)[mask, :][:, mask] # (original, original)
        dist_to_error = dist_to_error[mask] # (original, 1)

        # Vertices is always first in the input
        vertices = input[:, :3]
        edges = adjency_to_edge_list_torch_skip_diag(adj)

        config = data_utils.get_config()
        client, _, _ = data_utils.create_client(config)  
        cv = client.info.segmentation_cloudvolume(progress=False)
        root_without_num = int(root[:-4]) # Removing _000 for mesh retrieval
        mesh = cv.mesh.get(
            root_without_num, deduplicate_chunk_boundaries=False, remove_duplicate_vertices=False
        )[root_without_num]
        # seg_path = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1"
        # cv_seg = CloudVolume(seg_path, progress=False, use_https=True, parallel=True)
        # mesh = cv_seg.mesh.get(root_without_num, deduplicate_chunk_boundaries=False, remove_duplicate_vertices=True)[root_without_num]
        
        root_mesh = pv.make_tri_mesh(mesh.vertices, mesh.faces)

        is_proofread = data.get_is_proofread(root)
        num_initial_vertices = data.get_num_initial_vertices(root)

    return vertices, edges, labels, confidence, output, root_mesh, is_proofread, num_initial_vertices, dist_to_error