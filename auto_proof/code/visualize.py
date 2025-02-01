from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list
from auto_proof.code.model import create_model

import torch
import torch.nn as nn
import json
import pyvista as pv
import numpy as np
from cloudvolume import CloudVolume
# Remember to remove morphlink package that already exists in directory

CONFIG_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/base_config.json'

def get_root_output(model, device, data, idx):
    # For now it's just pulling a specific sample, later this will pull a specific sample in train/val/test
    sample = data.__getitem__(idx)
    input, labels, confidence, adj, root = sample 
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

    seg_path = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1"
    cv_seg = CloudVolume(seg_path, progress=False, use_https=True, parallel=True)
    mesh = cv_seg.mesh.get(root, deduplicate_chunk_boundaries=False, remove_duplicate_vertices=False)[root]
    root_mesh = pv.make_tri_mesh(mesh.vertices, mesh.faces)

    return vertices, edges, labels, confidence, output, root, root_mesh

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
        print("center of mass", center_of_mass)
    else:
        print("No 'Confident Merge Error' points found.")
        center_of_mass = vertices.mean(axis=0)  # Use center of all points as fallback

    
    
    plotter.subplot(0)
    plotter.add_mesh(skel_poly, color='black', )
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
    skel_poly_output = pv.PolyData(vertices_output, lines=lines)
    plotter.add_mesh(skel_poly_output, color='black')
    plotter.add_points(vertices_output, scalars=output, label='output', cmap='RdYlGn', point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'output'})
    # Doesn't work for interactive it seems
    plotter.add_text("Output", font_size=14) 
    plotter.link_views()

    plotter.subplot(2)
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()
    
    plotter.export_html(path)  

if __name__ == "__main__":    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    data = AutoProofDataset(config)

    model = create_model(config)
    ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20241104_132019/model_29'
    model.load_state_dict(torch.load(ckpt_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    root = 864691135654785218
    # root = 864691135463333789
    idx = data.get_root_index(root)
    # root = 864691135778235581
    path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/visualize_{root}_default_fov.html'
    vertices, edges, labels, confidence, output, root, root_mesh = get_root_output(model, device, data, idx)
    visualize(vertices, edges, labels, confidence, output, root_mesh, path)
