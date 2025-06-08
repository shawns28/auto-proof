from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list_torch_skip_diag
from auto_proof.code.model import create_model
from auto_proof.code.pre import data_utils

import torch
import torch.nn as nn
import json
import pyvista as pv
import numpy as np
import matplotlib as plt
# from cloudvolume import CloudVolume
from tqdm import tqdm
import joblib
from sklearn.decomposition import IncrementalPCA
from matplotlib.colors import LinearSegmentedColormap


def get_root_output(model, device, data, root):
    model.eval() # Marking this here due to async
    with torch.no_grad(): # Marking this here due to async
        # For now it's just pulling a specific sample, later this will pull a specific sample in train/val/test
        # model.to(device)
        idx = data.get_root_index(root)
        sample = data.__getitem__(idx)
        _ , input, labels, confidences, dist_to_error, rank , adj, mean_vertices = sample 
        input = input.float().to(device).unsqueeze(0) # (1, fov, d)
        labels = labels.float().to(device) # (fov, 1)
        confidences = confidences.float().to(device) # (fov, 1)
        adj = adj.float().to(device).unsqueeze(0) # (1, fov, fov)
        dist_to_error = dist_to_error.float().to(device) # (fov, 1)
        rank = rank.float().to(device) # (fov, 1)
        mean_vertices = mean_vertices.float().to(device) # (fov, 3)

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
        confidences = confidences[mask] # (original, 1)
        rank = rank[mask] # (original, 1)
        adj = adj.squeeze(0)[mask, :][:, mask] # (original, original)
        dist_to_error = dist_to_error[mask] # (original, 1)

        # Vertices is always first in the input
        vertices = input[:, :3]
        vertices = vertices + mean_vertices
        radius = input[:, 3:4]
        map_pe = input[:, 4:36]
        segclr_emb = input[:, 36:-1]
        has_segclr_emb = input[:, -1:]
        edges = adjency_to_edge_list_torch_skip_diag(adj)

        config = data_utils.get_config('client')
        client, _, _, _ = data_utils.create_client(config)  
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

    return vertices, edges, labels, confidences, output, root_mesh, is_proofread, num_initial_vertices, dist_to_error, segclr_emb, has_segclr_emb, radius, rank, map_pe

# Not detaching here because it causes weird error where it wants it to 
def visualize(vertices, edges, labels, confidences, output, root_mesh, dist_to_error, max_dist, show_tol, rank, box_cutoff, path):
    pv.set_jupyter_backend('trame')
    vertices = vertices.detach().cpu().numpy()
    edges = edges.detach().cpu().numpy()
    rank = rank.squeeze(-1).detach().cpu().numpy()

    vertex_is_valid = rank < box_cutoff
    edge_v1_indices = edges[:, 0]
    edge_v2_indices = edges[:, 1]
    v1_valid = vertex_is_valid[edge_v1_indices]
    v2_valid = vertex_is_valid[edge_v2_indices]
    edge_is_valid = v1_valid & v2_valid
    edge_is_non_valid = ~edge_is_valid
    valid_edges = edges[edge_is_valid]
    # Create the lines array for valid edges in VTK format
    valid_lines = np.column_stack([np.full(len(valid_edges), 2), valid_edges]).ravel()

    non_valid_edges = edges[edge_is_non_valid]
    non_valid_lines = np.column_stack([np.full(len(non_valid_edges), 2), non_valid_edges]).ravel()
    
    skel_poly_valid = pv.PolyData(vertices, lines=valid_lines)
    skel_poly_non_valid = pv.PolyData(vertices, lines=non_valid_lines)
    
    labels = labels.squeeze(-1).detach().cpu().numpy() # (original, )
    confidences = confidences.squeeze(-1).detach().cpu().numpy() # (original, )
    output = output.squeeze(-1).detach().cpu().numpy() # (original, 1)
    dist_to_error = dist_to_error.squeeze(-1).detach().cpu().numpy() # (original, )

    combined = labels * 2 + confidences
    c_m = (combined == 3).astype(bool)
    nc_nm = (combined == 0).astype(bool)
    c_nm = (combined == 1).astype(bool)
    # 0 = non-confident merge (This can't happen)
    # 1 = confident merge
    # 2 = non-confident non-merge
    # 3 = confident non-merge

    if show_tol:
        dist_mask_inc = np.logical_and(dist_to_error > 0, dist_to_error <= max_dist)
        dist_mask_exc = np.logical_or(dist_to_error == 0, dist_to_error > max_dist)
    
    pv.global_theme.multi_rendering_splitting_position = 0.5

    pv.start_xvfb()
    plotter = pv.Plotter(shape="1/2", off_screen=True, border=True, border_color='black')
    # All labels don't work for interactive
    plotter.add_title("Visualization")

    confident_merge_indices = np.where(combined == 3)[0] 
    confident_merge_points = vertices[confident_merge_indices] 

    # Calculate center of mass
    if len(confident_merge_points) > 0:
        center_of_mass = confident_merge_points.mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0)  # Use center of all points as fallback
    
    plotter.subplot(0)
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_valid, color='red')
    
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_non_valid, color='black')

    color_hex = "#B40426"  # Hexadecimal code for deep red
    deep_red = pv.Color(color_hex)
    color_hex = "#869BBF"  # Hexadecimal code for slightly grayish blue
    light_blue = pv.Color(color_hex)
    color_hex = "#3B4CC0"  # Hexadecimal code for deep blue
    deep_blue = pv.Color(color_hex)
    color_hex = "#FFD580" # Hexadecimal code for light orange
    light_orange = pv.Color(color_hex)

    if show_tol:
        if len(vertices[c_m & dist_mask_exc]) > 0:
            
            plotter.add_points(vertices[c_m & dist_mask_exc], color=deep_red, label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
        if len(vertices[nc_nm & dist_mask_exc]) > 0:
            
            plotter.add_points(vertices[nc_nm & dist_mask_exc], color=light_blue, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
        if len(vertices[c_nm & dist_mask_exc]) > 0:
            plotter.add_points(vertices[c_nm & dist_mask_exc], color=deep_blue, label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
        if len(vertices[dist_mask_inc]) > 0:
            plotter.add_points(vertices[dist_mask_inc], color=light_orange, label='tolerance points', point_size=10, render_points_as_spheres=True, show_vertices=True)
    else:
        if len(vertices[c_m]) > 0: 
            plotter.add_points(vertices[c_m], color=deep_red, label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
        if len(vertices[nc_nm]) > 0:
            plotter.add_points(vertices[nc_nm], color=light_blue, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
        if len(vertices[c_nm]) > 0:
            plotter.add_points(vertices[c_nm], color=deep_blue, label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    # Doesn't work for interactive
    plotter.add_legend()
    plotter.add_text("Labels", font_size=14)

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True

    plotter.subplot(1)
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_valid, color='red')
    
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_non_valid, color='black')

    vertices = np.concatenate((vertices, np.zeros((1, 3)), np.ones((1, 3))), axis=0)
    output = np.append(output, [0, 1])


    cmap_name = 'coolwarm'
    cmap = plt.colormaps.get_cmap(cmap_name)

    my_threshold = 0.05
    cmap = create_custom_threshold_red_colormap(threshold=my_threshold)

    plotter.add_points(vertices, scalars=output, label='predictions', cmap=cmap, point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'predictions'})
    # Doesn't work for interactive it seems
    plotter.add_text("Output", font_size=14) 
    plotter.link_views()

    plotter.subplot(2)
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()
    
    plotter.export_html(path)  
            
def visualize_pres(vertices, edges, labels, confidences, output, root_mesh, dist_to_error, max_dist, show_tol, rank, box_cutoff, path):
    pv.set_jupyter_backend('trame')
    vertices = vertices.detach().cpu().numpy()
    edges = edges.detach().cpu().numpy()
    rank = rank.squeeze(-1).detach().cpu().numpy()

    vertex_is_valid = rank < box_cutoff
    edge_v1_indices = edges[:, 0]
    edge_v2_indices = edges[:, 1]
    v1_valid = vertex_is_valid[edge_v1_indices]
    v2_valid = vertex_is_valid[edge_v2_indices]
    edge_is_valid = v1_valid & v2_valid
    edge_is_non_valid = ~edge_is_valid
    valid_edges = edges[edge_is_valid]
    # Create the lines array for valid edges in VTK format
    valid_lines = np.column_stack([np.full(len(valid_edges), 2), valid_edges]).ravel()

    non_valid_edges = edges[edge_is_non_valid]
    non_valid_lines = np.column_stack([np.full(len(non_valid_edges), 2), non_valid_edges]).ravel()

    skel_poly_valid = pv.PolyData(vertices, lines=valid_lines)
    skel_poly_non_valid = pv.PolyData(vertices, lines=non_valid_lines)

    labels = labels.squeeze(-1).detach().cpu().numpy() # (original, )
    confidences = confidences.squeeze(-1).detach().cpu().numpy() # (original, )
    output = output.squeeze(-1).detach().cpu().numpy() # (original, 1)
    dist_to_error = dist_to_error.squeeze(-1).detach().cpu().numpy() # (original, )

    combined = labels * 2 + confidences
    c_m = (combined == 3).astype(bool)
    nc_nm = (combined == 0).astype(bool)
    c_nm = (combined == 1).astype(bool)
    # 0 = non-confident merge (This can't happen)
    # 1 = confident merge
    # 2 = non-confident non-merge
    # 3 = confident non-merge

    if show_tol:
        dist_mask_inc = np.logical_and(dist_to_error > 0, dist_to_error <= max_dist)
        dist_mask_exc = np.logical_or(dist_to_error == 0, dist_to_error > max_dist)

    pv.global_theme.multi_rendering_splitting_position = 0.5

    pv.start_xvfb()
    plotter = pv.Plotter(shape="1/2", off_screen=True, border=True, border_color='black')
    # All labels don't work for interactive
    plotter.add_title("Visualization")

    confident_merge_indices = np.where(combined == 3)[0]
    confident_merge_points = vertices[confident_merge_indices]

    # Calculate center of mass
    if len(confident_merge_points) > 0:
        center_of_mass = confident_merge_points.mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0)  # Use center of all points as fallback

    plotter.subplot(0)
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_valid, color='red')

    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_non_valid, color='black')

    color_hex = "#B40426"  # Hexadecimal code for deep red
    deep_red = pv.Color(color_hex)
    color_hex = "#869BBF"  # Hexadecimal code for slightly grayish blue
    light_blue = pv.Color(color_hex)
    color_hex = "#3B4CC0"  # Hexadecimal code for deep blue
    deep_blue = pv.Color(color_hex)
    color_hex = "#FFD580" # Hexadecimal code for light orange
    light_orange = pv.Color(color_hex)

    # Define the mask for valid points for the label plot
    valid_points_mask_for_label_plot = vertex_is_valid

    if show_tol:
        # Confident Merge (c_m) AND valid AND (dist_mask_exc)
        if len(vertices[c_m & valid_points_mask_for_label_plot & dist_mask_exc]) > 0:
            plotter.add_points(vertices[c_m & valid_points_mask_for_label_plot & dist_mask_exc], color=deep_red, label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
        # Non-Confident Non-Merge (nc_nm) AND valid AND (dist_mask_exc)
        if len(vertices[nc_nm & valid_points_mask_for_label_plot & dist_mask_exc]) > 0:
            plotter.add_points(vertices[nc_nm & valid_points_mask_for_label_plot & dist_mask_exc], color=light_blue, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
        # Confident Non-Merge (c_nm) AND valid AND (dist_mask_exc)
        if len(vertices[c_nm & valid_points_mask_for_label_plot & dist_mask_exc]) > 0:
            plotter.add_points(vertices[c_nm & valid_points_mask_for_label_plot & dist_mask_exc], color=deep_blue, label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
        # Tolerance points (dist_mask_inc) AND valid
        if len(vertices[dist_mask_inc & valid_points_mask_for_label_plot]) > 0:
            plotter.add_points(vertices[dist_mask_inc & valid_points_mask_for_label_plot], color=light_orange, label='tolerance points', point_size=10, render_points_as_spheres=True, show_vertices=True)
    else:
        # Confident Merge (c_m) AND valid
        if len(vertices[c_m & valid_points_mask_for_label_plot]) > 0:
            plotter.add_points(vertices[c_m & valid_points_mask_for_label_plot], color=deep_red, label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
        # Non-Confident Non-Merge (nc_nm) AND valid
        if len(vertices[nc_nm & valid_points_mask_for_label_plot]) > 0:
            plotter.add_points(vertices[nc_nm & valid_points_mask_for_label_plot], color=light_blue, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
        # Confident Non-Merge (c_nm) AND valid
        if len(vertices[c_nm & valid_points_mask_for_label_plot]) > 0:
            plotter.add_points(vertices[c_nm & valid_points_mask_for_label_plot], color=deep_blue, label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)

    # Now, add the explicitly invalid points in black.
    # These are the points where vertex_is_valid is FALSE.
    invalid_vertices_for_label_plot = vertices[~vertex_is_valid]
    if len(invalid_vertices_for_label_plot) > 0:
        plotter.add_points(
            invalid_vertices_for_label_plot,
            color='black',
            label='invalid points (rank >= cutoff)',
            point_size=5,
            render_points_as_spheres=True,
            show_vertices=True
        )

    # Doesn't work for interactive
    plotter.add_legend()
    plotter.add_text("Labels", font_size=14)

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True

    plotter.subplot(1)
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_valid, color='red')

    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_non_valid, color='black')


    valid_vertices_for_output = vertices[vertex_is_valid]
    valid_output_scalars = output[vertex_is_valid]
    invalid_vertices_for_output = vertices[~vertex_is_valid]

    vertices_for_output_plot = np.concatenate((vertices, np.zeros((1, 3)), np.ones((1, 3))), axis=0)
    output_for_output_plot = np.append(output, [0, 1])

    my_threshold = 0.05
    cmap = create_custom_threshold_red_colormap(threshold=my_threshold)

    # Add valid points with the colormap based on their 'output' scalar values
    if len(valid_vertices_for_output) > 0:
        plotter.add_points(
            valid_vertices_for_output,
            scalars=valid_output_scalars,
            label='predictions',
            cmap=cmap,
            point_size=10,
            render_points_as_spheres=True,
            show_vertices=True,
            scalar_bar_args={'title':'predictions'} # Ensure scalar bar shows full [0,1] range
        )

    # Add invalid points (rank >= box_cutoff) in black
    if len(invalid_vertices_for_output) > 0:
        plotter.add_points(
            invalid_vertices_for_output,
            color='black',
            label='invalid points (rank >= cutoff)',
            point_size=5,
            render_points_as_spheres=True,
            show_vertices=True
        )


    # Doesn't work for interactive it seems
    plotter.add_text("Output", font_size=14)
    plotter.link_views()

    plotter.subplot(2)
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()

    plotter.export_html(path)
def visualize_segclr(vertices, edges, segclr_nodes, too_far_small_radius_indices, too_far_large_radius_indices, path):
    pv.set_jupyter_backend('trame')
    lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    skel_poly = pv.PolyData(vertices, lines=lines)
    pv.start_xvfb()
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, border=True, border_color='black')
    
    plotter.subplot(0, 0)
    plotter.add_mesh(skel_poly, color='black')
    plotter.add_points(segclr_nodes, point_size=5, render_points_as_spheres=True, show_vertices=True)
    
    plotter.camera.tight()
    plotter.subplot_border_visible = True

    plotter.subplot(0, 1)
    plotter.add_mesh(skel_poly, color='black')
    
    in_range_points = vertices[np.logical_and(too_far_small_radius_indices == False, too_far_large_radius_indices == False)]
    out_small_range_points = vertices[too_far_small_radius_indices]
    out_large_range_points = vertices[too_far_large_radius_indices]
    
    if len(in_range_points):
        plotter.add_points(in_range_points, point_size=5, color='green', render_points_as_spheres=True, show_vertices=True)
    if len(out_small_range_points):
        plotter.add_points(out_small_range_points, point_size=5, color='orange', render_points_as_spheres=True, show_vertices=True)
    if len(out_large_range_points):
        plotter.add_points(out_large_range_points, point_size=5, color='red', render_points_as_spheres=True, show_vertices=True)
    plotter.link_views()
    
    plotter.export_html(path)

def visualize_pca_segclr(vertices, edges, labels, confidences, root_mesh, pca_segclr_percentile, has_emb, radius, output, rank, pca_pe, output_with_threshold, box_cutoff, threshold, path):
    
    pv.set_jupyter_backend('trame')
    lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    # print("edges", edges)
    # print("lines", lines)
    skel_poly = pv.PolyData(vertices, lines=lines)
    pv.start_xvfb()
    plotter = pv.Plotter(shape=(2, 3), off_screen=True, border=True, border_color='black')
    
    vertex_is_valid = rank < box_cutoff
    edge_v1_indices = edges[:, 0]
    edge_v2_indices = edges[:, 1]
    v1_valid = vertex_is_valid[edge_v1_indices]
    v2_valid = vertex_is_valid[edge_v2_indices]
    edge_is_valid = v1_valid & v2_valid
    edge_is_non_valid = ~edge_is_valid
    valid_edges = edges[edge_is_valid]
    # Create the lines array for valid edges in VTK format
    valid_lines = np.column_stack([np.full(len(valid_edges), 2), valid_edges]).ravel()

    non_valid_edges = edges[edge_is_non_valid]
    non_valid_lines = np.column_stack([np.full(len(non_valid_edges), 2), non_valid_edges]).ravel()
    
    skel_poly_valid = pv.PolyData(vertices, lines=valid_lines)
    skel_poly_non_valid = pv.PolyData(vertices, lines=non_valid_lines)
    
    cmap_name = 'coolwarm'
    cmap = plt.colormaps.get_cmap(cmap_name)


    vertices_output = np.concatenate((vertices, np.zeros((1, 3)), np.ones((1, 3))), axis=0)

    # Mesh
    plotter.subplot(0, 0)
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)

    combined = labels * 2 + confidences
    c_m = (combined == 3).astype(bool)
    nc_nm = (combined == 0).astype(bool)
    c_nm = (combined == 1).astype(bool)

    confident_merge_indices = np.where(combined == 3)[0] 
    confident_merge_points = vertices[confident_merge_indices] 

    # Calculate center of mass
    if len(confident_merge_points) > 0:
        center_of_mass = confident_merge_points.mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0)  # Use center of all points as fallback

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True

    # map pe
    plotter.subplot(0, 1)
    plotter.add_mesh(skel_poly, color='black')
    plotter.add_points(vertices, scalars=pca_pe, point_size=10, rgb=True, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'pe_pca'})
    plotter.link_views()

    # pca percentile
    plotter.subplot(0, 2)

    plotter.add_mesh(skel_poly, color='black')
    pca_segclr_percentile[has_emb == False] = np.ones(3)
    plotter.add_points(vertices, scalars=pca_segclr_percentile, point_size=10, rgb=True, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'segclr_pca'})

    # plotter.camera.tight()
    # plotter.subplot_border_visible = True
    plotter.link_views()

    # labels
    plotter.subplot(1, 0)

    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_valid, color='red')
    
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_non_valid, color='black')

    color_hex = "#B40426"  # Hexadecimal code for deep red
    deep_red = pv.Color(color_hex)
    color_hex = "#869BBF"  # Hexadecimal code for slightly grayish blue
    light_blue = pv.Color(color_hex)
    color_hex = "#3B4CC0"  # Hexadecimal code for deep blue
    deep_blue = pv.Color(color_hex)

    if len(vertices[c_m]) > 0:
        plotter.add_points(vertices[c_m], color=deep_red, label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[nc_nm]) > 0:
        plotter.add_points(vertices[nc_nm], color=light_blue, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[c_nm]) > 0:
        plotter.add_points(vertices[c_nm], color=deep_blue, label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    # Doesn't work for interactive
    # plotter.add_legend()
    # plotter.add_text("Labels", font_size=14)
    plotter.link_views()

    # output
    plotter.subplot(1, 2) 

    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_valid, color='red')
    
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_non_valid, color='black')

    output = np.append(output, [0., 1.])

    plotter.add_points(vertices_output, scalars=output, label='predictions', cmap=cmap, point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'predictions'})
    # Doesn't work for interactive it seems
    plotter.add_text("Output", font_size=14) 
    plotter.link_views()

    # Threshold

    plotter.subplot(1, 1) 

    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_valid, color='red')
    
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_non_valid, color='black')

    # output = np.append(output, [0, 1])
    # print("output", output)
    # output_t= np.where(output > threshold, 1., 0.)
    # print(output_t)
    output_with_threshold = np.append(output_with_threshold, [0., 1.])

    plotter.add_points(vertices_output, scalars=output_with_threshold, label='threshold', cmap=cmap, point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':f'threshold at {threshold}'})
    # Doesn't work for interactive it seems
    plotter.add_text("Output", font_size=14) 
    plotter.link_views()

    plotter.export_html(path)

def create_custom_threshold_red_colormap(threshold=0.05, name='blue_lightred_red_threshold'):
    """
    Creates a colormap that is deep blue at 0, gradients to a very light blue
    at the specified threshold, instantly shifts to a light red at the threshold,
    and then gradients to deep red at 1.

    Parameters
    ----------
    threshold : float
        The value (between 0 and 1) where the colormap transitions from
        blue to a light blue, then instantly shifts to light red, and
        continues to deep red.
    name : str
        The name of the colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A Matplotlib colormap object.
    """
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")

    # Get the deep blue and deep red from the 'coolwarm' colormap
    cmap_coolwarm_original = plt.pyplot.get_cmap('coolwarm')
    deep_blue = cmap_coolwarm_original(0.0) # (R, G, B, A) tuple for deep blue
    deep_red = cmap_coolwarm_original(1.0)  # (R, G, B, A) tuple for deep red

    # Define a "really light blue" color in RGB (0-1)
    # This color has components of Green and Blue to make it "light" (desaturated)
    light_blue = (0.7, 0.7, 1.0) # R, G, B values (0-1)

    # Define a "really light red" color in RGB (0-1)
    light_red = (1.0, 0.7, 0.7) # R, G, B values (0-1)

    # cdict defines the interpolation for each color channel (Red, Green, Blue)
    # The structure for each channel is:
    # [(position, value_at_start_of_segment, value_at_end_of_segment), ...]
    # A sharp transition at 'position' occurs if value_at_start_of_segment != value_at_end_of_segment
    cdict = {
        'red':   [[0.0, deep_blue[0], deep_blue[0]],          # From 0.0 to threshold, red component gradients from deep_blue[0] to light_blue[0]
                  [threshold, light_blue[0], light_red[0]],  # At threshold, there's a sharp jump: value is light_blue[0] from left, light_red[0] from right
                  [1.0, deep_red[0], deep_red[0]]],           # From threshold to 1.0, red component gradients from light_red[0] to deep_red[0]

        'green': [[0.0, deep_blue[1], deep_blue[1]],          # From 0.0 to threshold, green component gradients from deep_blue[1] to light_blue[1]
                  [threshold, light_blue[1], light_red[1]],  # At threshold, sharp jump from light_blue[1] to light_red[1]
                  [1.0, deep_red[1], deep_red[1]]],           # From threshold to 1.0, green component gradients from light_red[1] to deep_red[1]

        'blue':  [[0.0, deep_blue[2], deep_blue[2]],          # From 0.0 to threshold, blue component gradients from deep_blue[2] to light_blue[2]
                  [threshold, light_blue[2], light_red[2]],  # At threshold, sharp jump from light_blue[2] to light_red[2]
                  [1.0, deep_red[2], deep_red[2]]]            # From threshold to 1.0, blue component gradients from light_red[2] to deep_red[2]
    }

    return LinearSegmentedColormap(name, cdict)

def visualize_whole_cell(vertices, edges, labels, confidences, output, mesh, threshold, save_path):
    pv.set_jupyter_backend('trame')
    lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    
    skel_poly = pv.PolyData(vertices, lines=lines)

    combined = labels * 2 + confidences
    c_m = (combined == 3).astype(bool)
    nc_nm = (combined == 0).astype(bool)
    c_nm = (combined == 1).astype(bool)
    # 0 = non-confident merge (This can't happen)
    # 1 = confident merge
    # 2 = non-confident non-merge
    # 3 = confident non-merge
    
    pv.global_theme.multi_rendering_splitting_position = 0.5

    pv.start_xvfb()
    plotter = pv.Plotter(shape="1/2", off_screen=True, border=True, border_color='black')
    # All labels don't work for interactive
    plotter.add_title("Visualization")

    confident_merge_indices = np.where(combined == 3)[0] 
    confident_merge_points = vertices[confident_merge_indices] 

    # Calculate center of mass
    if len(confident_merge_points) > 0:
        center_of_mass = confident_merge_points.mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0)  # Use center of all points as fallback
    
    plotter.subplot(0)
    plotter.add_mesh(skel_poly, color='black')

    color_hex = "#B40426"  # Hexadecimal code for deep red
    deep_red = pv.Color(color_hex)
    color_hex = "#869BBF"  # Hexadecimal code for slightly grayish blue
    light_blue = pv.Color(color_hex)
    color_hex = "#3B4CC0"  # Hexadecimal code for deep blue
    deep_blue = pv.Color(color_hex)
    color_hex = "#FFD580" # Hexadecimal code for light orange
    light_orange = pv.Color(color_hex)


    if len(vertices[c_m]) > 0: 
        plotter.add_points(vertices[c_m], color=deep_red, label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[nc_nm]) > 0:
        plotter.add_points(vertices[nc_nm], color=light_blue, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[c_nm]) > 0:
        plotter.add_points(vertices[c_nm], color=deep_blue, label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    # Doesn't work for interactive
    plotter.add_legend()
    plotter.add_text("Labels", font_size=14)

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True

    plotter.subplot(1)
    plotter.add_mesh(skel_poly, color='black')

    vertices_output = vertices.copy()
    vertices_output = np.concatenate((vertices_output, np.zeros((1, 3)), np.ones((1, 3))), axis=0)
    output = np.append(output, [0, 1])

    # cmap_name = 'coolwarm'
    # cmap = plt.colormaps.get_cmap(cmap_name)

    cmap = create_custom_threshold_red_colormap(threshold=threshold)

    plotter.add_points(vertices_output, scalars=output, label='predictions', cmap=cmap, point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'predictions'})
    # Doesn't work for interactive it seems
    plotter.add_text("Output", font_size=14) 
    plotter.link_views()

    plotter.subplot(2)
    plotter.add_mesh(mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()
    
    plotter.export_html(save_path)  

if __name__ == "__main__":
    ckpt_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/ckpt/"   
    # run_id = 'AUT-215'
    # run_id = 'AUT-255'
    run_id = 'AUT-272'
    run_id = 'AUT-275' # First segclr
    run_id = 'AUT-301' # best segclr so far
    run_id = 'AUT-321'
    run_id = 'AUT-330'
    # run_id = 'AUT-331'
    # run_id = 'AUT-302'
    # epoch = 40
    epoch = 60
    run_dir = f'{ckpt_dir}{run_id}/'
    # TODO: Uncomment below after segclr testing
    with open(f'{run_dir}config.json', 'r') as f:
        config = json.load(f)
    # with open('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/base_config.json', 'r') as f:
    #     config = json.load(f)
    # config['loader']['relative_vertices'] = True
    # config['loader']['zscore_radius'] = False
    # config['loader']['zscore_segclr'] = False
    # config['loader']['l2_norm_segclr'] = False
    # config['loader']['zscore_pe'] = False
    config['data']['all_roots'] = "all_roots_og.txt"
    config['data']['train_roots'] = "train_roots_og.txt"
    config['data']['val_roots'] = "val_roots_og.txt"
    config['data']['test_roots'] = "test_roots_og.txt"

    if run_id == 'AUT-330':
        config['data']['labels_dir'] = "labels_at_1300_ignore_inbetween/"

    data = AutoProofDataset(config, 'val')
    # config['model']['depth'] = 3
    # config['model']['n_head'] = 4
    model = create_model(config)
    # ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20241104_132019/model_29'
    # ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20250212_125759/model_1.pt'
    # ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20250215_212005/model_34.pt'
    # ckpt_path = f'{run_dir}model_55.pt'
    ckpt_path = f'{run_dir}model_{epoch}.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # root = 864691136443843459
    # root = 864691134884807418
    # root = 864691135654785218 # in train
    # root = 864691135463333789 # in train
    # idx = data.get_root_index(root)
    # roots = [864691135778235581] # in train
    # roots = [864691135463333789, 864691135778235581, 864691135463333789, 864691135778235581]
    # roots = [864691135463333789, 864691135778235581, 864691135463333789, 864691135778235581]
    # roots = [864691135937424949]
    # roots = ['864691136041340246_000']
    # roots = ['864691135463333789_000']
    # roots = ['864691135439772402_000']
    # roots = ['864691135191257833_000']
    # roots = ['864691135490886887_000']
    # roots = ['864691136521643153_000']
    # roots = [864691135373853640, 864691135658276738, 864691135684548279, 864691135684548023, 864691135404040430, 864691135065359940, 864691135386905985, 864691135404090350, 864691135396807713, 864691135387767681, 864691135341705649, 864691135446000274, 864691135947108705, 864691135947108193, 864691135447045460, 864691135359346776, 864691135987462147]
    # roots = [864691135684548279]
    # roots = [864691135396807713]
    # roots = [864691135989085184, 864691135683792114, 864691136483519276, 864691135571609125, 864691135876419923, 864691136310246234, 864691136009015212, 864691135697685269, 864691136266911476]
    # Roots that are sus because conf and error are similar but not exact
    # roots = [864691134940047459, 864691135411419697, 864691135724417451, 864691135472111666]
    # roots = ['864691135571361317_000', '864691135991989185_000', '864691136952448223_000', '864691136296738587_000', '864691135855823662_000', '864691136619551117_000', '864691136558482594_000', '864691136388405623_000']
    # roots = ['864691135472212274_000', '864691135615918697_000', '864691136736838510_000', '864691136286727107_000', '864691135538252914_000', '864691135517565322_000', '864691135987088387_000', '864691135952189475_000']
    max_dist = config['trainer']['max_dist']
    # stuff = [(model, device, data, 864691135463333789), (model, device, data, 864691135778235581), (model, device, data, 864691135463333789), (model, device, data, 864691135778235581)]
    # num_processes = len(stuff)
    # with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(stuff)) as pbar:
    #     for root in pool.imap_unordered(visualize_root, stuff):
    #         pbar.update()
    config['trainer']['visualize_cutoff'] = 4000
    config['trainer']['branch_degrees'] = [3, 4, 5]
    box_cutoff = config['data']['box_cutoff']
    config['loader']['fov'] = 500
    # config['trainer']['show_tol'] = False
    # roots = ['864691135463333789_000']
    # roots = ['864691135778235581_000']
    # roots = ['864691136227494225_000', '864691136227495761_000', '864691136912265969_000', '864691136926085706_000', '864691135777645664_000', '864691136619433869_002']
    # roots = ['864691135361128519_000', '864691135374663666_000', '864691135387215745_000', '864691135395011829_000', '864691135441162824_000', '864691135447598932_000', '864691135463303486_000', '864691135476697512_000']
    # roots = ['864691134940888163_000']
    # roots = ['864691136437067166_000', '864691136662768094_000', '864691136578169108_000', '864691136990572949_000', '864691136974211868_000', '864691134989119098_000', '864691135082409975_000']
    # roots = ['864691135324927034_000', '864691135327500274_000', '864691135328621376_000', '864691135335739881_000', '864691135335745769_000']
    # roots = ['864691134723510360_000', '864691134917390602_000', '864691135118894813_000', '864691135122469671_000', '864691135125642257_000', '864691135187356435_000', '864691135361185863_000', '864691135572385061_000', '864691135635398115_000', '864691135657412579_000', '864691135682085268_000', '864691135884404592_000']
    # roots = ['864691135657412579_000']
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/missed_roots_301.txt")
    # roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/val_conf_no_error_in_box_roots_og.txt")
    roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/before_sven/intersect_nosegclr.txt")
    # roots = roots[:2]
    roots = np.random.choice(roots, size=20, replace=False)
    roots = ['864691136380179925_000']
    roots = ['864691135101202976_000', '864691135645858543_000', '864691135698811781_000', '864691136239194172_000', '864691136889944242_000']
    roots = ['864691135187356435_000']
    roots = ['864691135594727723_000']
    
    #roots = ['864691135864976604_000']
    # print("roots len", len(roots))
    # roots = ['864691134940888163_000'] # where segclr missed some embeddings
    # roots = ['864691135187356435_000'] # obvious segclr difference
    # roots = ['864691135187356435_000', '864691135341835953_000', '864691135257060527_000', '864691135272408593_000', '864691135275761637_000']
    # roots = ['864691135066482244_000'] # The one with an edge error
    # roots = ['864691135572385061_000'] # proofread one
    # roots = ['864691135187356435_000', '864691135572385061_000', '864691134940888163_000']
    # roots = ['864691135682085268_000'] # one that my model usually misses
    # for i in range(10):
    # loaded_ipca = joblib.load("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/segclr_pca.joblib")
    # scale_max = np.load('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/pca_total_max.npy')
    # scale_min = np.load('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/pca_total_min.npy')
    # scale max [  71.32926664 3114.10755037  887.24305467]
    # scale min [-44831.26359002  -1819.20141471  -3972.91196383]
    
    # scale_max = np.full(3, 25)
    # scale_min = np.full(3, -25)
    # print("scale max", scale_max)
    # print("scale min", scale_min)
    # range_vals = scale_max - scale_min
    # range_vals[range_vals == 0] = 1.0 # Avoid division by zero, component will map to 0.5 or 0

    # segclr_mean = np.load("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/segclr_mean.npy")
    # segclr_std = np.load("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/segclr_std.npy")
    # pos_mean = np.load("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/pos_mean.npy")
    # pos_std = np.load("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/pos_std.npy")


    for root in roots:
        # try:
        # root = data.get_random_root()
        # while data.get_num_initial_vertices(root) > config['trainer']['visualize_cutoff']:
        #     root = data.get_random_root()
        num_initial_vertices = data.get_num_initial_vertices(root)
        print("num_initial_vertice", num_initial_vertices, "for root", root)
        if num_initial_vertices < config['trainer']['visualize_cutoff']:
            print("getting root output")
            vertices, edges, labels, confidences, output, root_mesh, is_proofread, num_initial_vertices, dist_to_error, segclr_emb, has_segclr_emb, radius, rank, pe = get_root_output(model, device, data, root)
            # print(len(vertices))
            # print(len(labels))
            # print("labels", labels)
            # print("confidences", confidences)
            print("is_proofread", is_proofread)
            
            # path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/before_pres/{root}.html'
            # print("visualizing root")
            # visualize(vertices, edges, labels, confidences, output, root_mesh, dist_to_error, max_dist, config['trainer']['show_tol'], rank, box_cutoff, path)
            
            path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/before_pres/{root}.html'
            print("visualizing pres root")
            visualize_pres(vertices, edges, labels, confidences, output, root_mesh, dist_to_error, max_dist, config['trainer']['show_tol'], rank, box_cutoff, path)

            # path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/segclr_test/{root}_segclr.html'
            # visualize_segclr(vertices, edges, path)
            
            # path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/testing_colors/{root}_inputs.html'

            # vertices = vertices.detach().cpu().numpy()
            # edges = edges.detach().cpu().numpy()
            # labels = labels.squeeze(-1).detach().cpu().numpy() # (original, )
            # confidences = confidences.squeeze(-1).detach().cpu().numpy() # (original, )
            # segclr_emb = segclr_emb.detach().cpu().numpy()
            # has_segclr_emb = has_segclr_emb.squeeze(-1).detach().cpu().numpy()
            # radius = radius.squeeze(-1).detach().cpu().numpy()
            # rank = rank.squeeze(-1).detach().cpu().numpy()
            # # print("segclr emb", segclr_emb)
            # output = output.squeeze(-1).detach().cpu().numpy()
            # pe = pe.detach().cpu().numpy()
 
            # # pe = (pe - pos_mean) / pos_std
            # pe_pca = IncrementalPCA(n_components=3)
            # pe_pca.fit(pe)
            # pe_rgb = pe_pca.transform(pe)
            # pe_max = pe_rgb.max(axis=0)
            # pe_min = pe_rgb.min(axis=0)
            # range_vals = pe_max - pe_min
            # range_vals[range_vals == 0] = 1.0
            # pe_scaled = (pe_rgb - pe_min) / range_vals
            # pe_clipped = np.clip(pe_scaled, 0, 1)

            # # segclr_emb = (segclr_emb - segclr_mean) / segclr_std
            # ipca = IncrementalPCA(n_components=3)
            # ipca.fit(segclr_emb)
            # # print(segclr_rgb)

            # segclr_rgb = ipca.transform(segclr_emb)
            # root_max = segclr_rgb.max(axis=0)
            # root_min = segclr_rgb.min(axis=0)
            # # print("root max", root_max)
            # # print("root min", root_min)
            # range_vals = root_max - root_min
            # range_vals[range_vals == 0] = 1.0
            # segclr_scaled = (segclr_rgb - root_min) / range_vals
            # segclr_clipped = np.clip(segclr_scaled, 0, 1)
            # percentile_95 = np.percentile(segclr_rgb, 95, axis=0)
            # percentile_5 = np.percentile(segclr_rgb, 5, axis=0)
            
            # # print("max 95", percentile_95)
            # # print("min 5", percentile_5)
            # range_vals = percentile_95 - percentile_5
            # range_vals[range_vals == 0] = 1.0
            # segclr_scaled_percentile = (segclr_rgb - percentile_5) / range_vals
            # segclr_clipped_percentile = np.clip(segclr_scaled_percentile, 0, 1)

            # # segclr_scaled = (segclr_rgb - scale_min) / range_vals
            # # print(segclr_scaled)
            # # print("segclr scaled", segclr_clipped)
            # threshold = 0.05
            # output_with_threshold = np.where(output > threshold, 1., 0.)
            # print("visualizing inputs")
            # visualize_pca_segclr(vertices, edges, labels, confidences, root_mesh, segclr_clipped_percentile, has_segclr_emb, radius, output, rank, pe_clipped, output_with_threshold, box_cutoff,threshold,  path)
        # except Exception as e:
        #     print("Failed visualization for root id: ", root, "error: ", e)
        #     continue

    # roots = [864691136831441518, 864691136379030485, 864691135585405308, 864691135278463137, 864691136673368967, 864691136084808300, 864691135976247107, 864691135946876001, 864691135725734719, 864691135740501099, 864691135772003147, 864691136024491577]
    # for root in roots:
    #     print("root", root, "num initial", data.get_num_initial_vertices(root))