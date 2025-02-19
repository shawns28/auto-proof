from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list
from auto_proof.code.pre import data_utils
from auto_proof.code.model import create_model

import torch
import torch.nn as nn
import json
import pyvista as pv
import numpy as np
from cloudvolume import CloudVolume
from tqdm import tqdm
import multiprocessing
# Remember to remove morphlink package that already exists in directory

CONFIG_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/base_config.json'

def get_root_output(model, device, data, root):
    model.eval() # Marking this here due to async
    with torch.no_grad(): # Marking this here due to async
        # For now it's just pulling a specific sample, later this will pull a specific sample in train/val/test
        # model.to(device)
        idx = data.get_root_index(root)
        sample = data.__getitem__(idx)
        input, labels, confidence, adj = sample 
        input = input.float().to(device).unsqueeze(0) # (1, fov, d)
        labels = labels.float().to(device) # (fov, 1)
        confidence = confidence.float().to(device) # (fov, 1)
        adj = adj.float().to(device).unsqueeze(0) # (1, fov, fov)

        output = model(input, adj) # (1, fov, 2)
        # Still need sigmoid here since we're not doing cross entropy loss since we're not taking the loss
        sigmoid = nn.Sigmoid()
        output = sigmoid(output) # (1, fov, 2)
        output = output.squeeze(0) # (fov, 2)

        # original represents the original points that weren't buffered
        mask = labels != -1
        mask = mask.squeeze(-1) # (original)
        
        # Apply mask to get original points
        output = output[mask] # (original, 2)
        input = input.squeeze(0)[mask] # (original, d)
        labels = labels[mask] # (original, 1)
        confidence = confidence[mask] # (original, 1)
        adj = adj.squeeze(0)[mask, :][:, mask] # (original, original)

        # Vertices is always first in the input
        vertices = input[:, :3]
        edges = adjency_to_edge_list(adj)

        config = data_utils.get_config()
        client, _, _ = data_utils.create_client(config)  
        cv = client.info.segmentation_cloudvolume(progress=False)
        mesh = cv.mesh.get(
            root, deduplicate_chunk_boundaries=False, remove_duplicate_vertices=False
        )[root]

        # seg_path = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1"
        # cv_seg = CloudVolume(seg_path, progress=False, use_https=True, parallel=True)
        # mesh = cv_seg.mesh.get(root, deduplicate_chunk_boundaries=False, remove_duplicate_vertices=True)[root]
        
        root_mesh = pv.make_tri_mesh(mesh.vertices, mesh.faces)

        is_proofread = data.get_is_proofread(root)
        num_initial_vertices = data.get_num_initial_vertices(root)

    return vertices, edges, labels, confidence, output, root_mesh, is_proofread, num_initial_vertices

def visualize(vertices, edges, labels, confidence, output, root_mesh, path):
    pv.set_jupyter_backend('trame')
    vertices = vertices.cpu().numpy()
    lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    skel_poly = pv.PolyData(vertices, lines=lines)

    labels = labels.squeeze(-1).cpu().numpy()
    confidence = confidence.squeeze(-1).cpu().numpy()
    output = output.squeeze(-1).detach().cpu().numpy()
    output = output[:, 1] # I think that the second column is non-merge probability
    combined = labels * 2 + confidence
    nc_m = (combined == 0).astype(bool)
    c_m = (combined == 1).astype(bool)
    nc_nm = (combined == 2).astype(bool)
    c_nm = (combined == 3).astype(bool)
    # combined[combined == 0] = 0.33
    # combined[combined == 1] = 0
    # combined[combined == 2] = 0.66
    # combined[combined == 3] = 1
    # 0 = non-confident merge (This can't happen)
    # 1 = confident merge
    # 2 = non-confident non-merge
    # 3 = confident non-merge
    
    pv.global_theme.multi_rendering_splitting_position = 0.5

    pv.start_xvfb()
    plotter = pv.Plotter(shape="1/2", off_screen=True, border=True, border_color='black')
    # All labels don't work for interactive
    plotter.add_title("Visualization")

    # Change this!
    confident_merge_indices = np.where(combined == 1)[0] 
    confident_merge_points = vertices[confident_merge_indices] 

    # Calculate center of mass
    if len(confident_merge_points) > 0:
        center_of_mass = confident_merge_points.mean(axis=0)
        # print("Center of mass", center_of_mass)
    else:
        center_of_mass = vertices.mean(axis=0)  # Use center of all points as fallback
        # print("No merge errors")
    
    plotter.subplot(0)
    # combined = np.append(combined, [0, 1])
    # vertices_copy = np.concatenate((vertices, np.full((1, 3), 1814336), np.full((1, 3), 1814336)), axis=0)
    plotter.add_mesh(skel_poly, color='black')
    # plotter.add_points(vertices_copy, scalars=combined, label='labels', cmap='RdYlGn', point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'labels'})
    # plotter.add_points(vertices, scalars=combined, label='labels', cmap='RdYlGn', point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'labels'})
    if len(vertices[nc_m]) > 0:
        plotter.add_points(vertices[nc_m], color='orange', label='non-confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[c_m]) > 0:
        plotter.add_points(vertices[c_m], color='red', label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[nc_nm]) > 0:
        color_hex = "#9ACD32"  # Hexadecimal code for yellow-green
        yellow_green = pv.Color(color_hex) 
        plotter.add_points(vertices[nc_nm], color=yellow_green, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[c_nm]) > 0:
        plotter.add_points(vertices[c_nm], color='green', label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    # Doesn't work for interactive
    plotter.add_legend()
    plotter.add_text("Labels", font_size=14)

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True


    plotter.subplot(1)
    vertices_output = vertices.copy()
    vertices_output = np.concatenate((vertices_output, np.zeros((1, 3)), np.ones((1, 3))), axis=0)
    output = np.append(output, [0, 1])
    skel_poly_output = pv.PolyData(vertices_output, lines=lines)
    plotter.add_mesh(skel_poly_output, color='black')
    plotter.add_points(vertices_output, scalars=output, label='predictions', cmap='RdYlGn', point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'predictions'})
    # Doesn't work for interactive it seems
    plotter.add_text("Output", font_size=14) 
    plotter.link_views()

    plotter.subplot(2)
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()
    
    plotter.export_html(path)  

def visualize_root(stuff):
    try:
        model, device, data, root = stuff
        path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/visualize_{root}_ckpt34_500_fov.html'
        vertices, edges, labels, confidence, output, root_mesh, is_proofread, num_intitial_vertices = get_root_output(model, device, data, root)
        print("is_proofread", is_proofread)
        print("num_intitial_vertice", num_intitial_vertices)
        visualize(vertices, edges, labels, confidence, output, root_mesh, path)
    except Exception as e:
        print("Failed visualization for root id: ", root, "error: ", e)
            

if __name__ == "__main__":    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    data = AutoProofDataset(config, 'train')

    model = create_model(config)
    # ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20241104_132019/model_29'
    # ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20250212_125759/model_1.pt'
    ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20250215_212005/model_34.pt'
    model.load_state_dict(torch.load(ckpt_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # always cpu
    device = "cpu"
    model.to(device)
    # model.eval()

    # root = 864691136443843459
    # root = 864691134884807418
    # root = 864691135654785218 # in train
    # root = 864691135463333789 # in train
    # idx = data.get_root_index(root)
    # root = 864691135778235581 in train
    # roots = [864691135463333789, 864691135778235581, 864691135463333789, 864691135778235581]
    # roots = [864691135463333789, 864691135778235581, 864691135463333789, 864691135778235581]

    stuff = [(model, device, data, 864691135463333789), (model, device, data, 864691135778235581), (model, device, data, 864691135463333789), (model, device, data, 864691135778235581)]
    num_processes = len(stuff)
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(stuff)) as pbar:
        for root in pool.imap_unordered(visualize_root, stuff):
            pbar.update()

    # for root in roots:
    #     try:
    #         path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/visualize_{root}_ckpt34_500_fov.html'
    #         vertices, edges, labels, confidence, output, root_mesh, is_proofread, num_intitial_vertices = get_root_output(model, device, data, root)
    #         print("is_proofread", is_proofread)
    #         print("num_intitial_vertice", num_intitial_vertices)
    #         visualize(vertices, edges, labels, confidence, output, root_mesh, path)
    #     except Exception as e:
    #         print("Failed visualization for root id: ", root, "error: ", e)
    #         continue

    # roots = [864691136831441518, 864691136379030485, 864691135585405308, 864691135278463137, 864691136673368967, 864691136084808300, 864691135976247107, 864691135946876001, 864691135725734719, 864691135740501099, 864691135772003147, 864691136024491577]
    # for root in roots:
    #     print("root", root, "num initial", data.get_num_initial_vertices(root))