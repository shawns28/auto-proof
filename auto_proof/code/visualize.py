from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list_torch_skip_diag
from auto_proof.code.model import create_model
from auto_proof.code.pre import data_utils

import torch
import torch.nn as nn
import json
import pyvista as pv
import numpy as np
import matplotlib as plt
from tqdm import tqdm
import joblib
from sklearn.decomposition import IncrementalPCA
from matplotlib.colors import LinearSegmentedColormap

def get_root_output(model, device, data, root):
    """Retrieves and processes the model output for a given root.

    Args:
        model (torch.nn.Module): The neural network model to use for inference.
        device (torch.device): The device (CPU or GPU) to run the model on.
        data (torch.utils.data.Dataset): The dataset object containing the data.
        root (str): The identifier for the specific root to process (e.g., 'root_id_000').

    Returns:
        tuple: A tuple containing:
            - vertices (torch.Tensor): The 3D coordinates of the original points.
            - edges (torch.Tensor): The list of edges connecting the original points.
            - labels (torch.Tensor): The ground truth labels for the original points.
            - confidences (torch.Tensor): The confidence scores for the original points.
            - output (torch.Tensor): The model's predicted output for the original points
              (after sigmoid and padding masking).
            - root_mesh (pyvista.PolyData): The 3D mesh object for the given root.
            - is_proofread (bool): A flag indicating if the root has been proofread.
            - num_initial_vertices (int): The number of initial vertices before buffering.
            - dist_to_error (torch.Tensor): The distance to error for original points.
            - segclr_emb (torch.Tensor): SegCLR embeddings for original points.
            - has_segclr_emb (torch.Tensor): Flag indicating if SegCLR emb at point exists.
            - radius (torch.Tensor): Radius information for original points.
            - rank (torch.Tensor): Rank information for original points.
            - map_pe (torch.Tensor): Positional encoding for original points.
    """
    model.eval()
    with torch.no_grad():
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
        
        root_mesh = pv.make_tri_mesh(mesh.vertices, mesh.faces)

        is_proofread = data.get_is_proofread(root)
        num_initial_vertices = data.get_num_initial_vertices(root)

    return vertices, edges, labels, confidences, output, root_mesh, is_proofread, num_initial_vertices, dist_to_error, segclr_emb, has_segclr_emb, radius, rank, map_pe

def visualize(vertices, edges, labels, confidences, output, root_mesh, dist_to_error, max_dist, show_tol, rank, box_cutoff, path):
    """Visualizes the skeleton, mesh, and model predictions using PyVista.

    This function generates a multi-panel plot showing:
    1. Skeleton points colored by ground truth labels and confidence.
    2. Skeleton points colored by model predictions.
    3. The 3D root mesh.

    It uses coolwarm for the predictions with red being an error.
    The core box predictions are highlighted by red edges.

    Args:
        vertices (torch.Tensor): The 3D coordinates of the skeleton points.
        edges (torch.Tensor): The connectivity (edges) of the skeleton.
        labels (torch.Tensor): Ground truth labels for each vertex.
        confidences (torch.Tensor): Confidence scores for each vertex.
        output (torch.Tensor): Model's predicted output (probabilities) for each vertex.
        root_mesh (pyvista.PolyData): The 3D mesh object of the root.
        dist_to_error (torch.Tensor): Distance to error for each vertex.
        max_dist (float): Maximum distance for tolerance visualization.
        show_tol (bool): If True, show points within `max_dist` as 'tolerance points'.
        rank (torch.Tensor): Rank of each vertex, used for `box_cutoff`.
        box_cutoff (float): Threshold for `rank` to determine valid vertices/edges.
        path (str): The file path to save the generated HTML visualization.
    """
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

    plotter.add_points(vertices, scalars=output, label='predictions', cmap=cmap, point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'predictions'})

    plotter.link_views()

    plotter.subplot(2)
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()
    
    plotter.export_html(path)  
            
def visualize_pres(vertices, edges, labels, confidences, output, root_mesh, dist_to_error, max_dist, show_tol, rank, box_cutoff, path):
    """Visualizes the skeleton, mesh, and model predictions using PyVista.

    This function generates a multi-panel plot showing:
    1. Skeleton points colored by ground truth labels and confidence.
    2. Skeleton points colored by model predictions.
    3. The 3D root mesh.

    It uses a custom red and blue colormap based off of a given threshold
    for the predictions with red being an error and the shift to blue at the threshold.
    The core box predictions have the above colormap and the rest of the nodes
    are black and smaller in size.

    Args:
        vertices (torch.Tensor): The 3D coordinates of the skeleton points.
        edges (torch.Tensor): The connectivity (edges) of the skeleton.
        labels (torch.Tensor): Ground truth labels for each vertex.
        confidences (torch.Tensor): Confidence scores for each vertex.
        output (torch.Tensor): Model's predicted output (probabilities) for each vertex.
        root_mesh (pyvista.PolyData): The 3D mesh object of the root.
        dist_to_error (torch.Tensor): Distance to error for each vertex.
        max_dist (float): Maximum distance for tolerance visualization.
        show_tol (bool): If True, show points within `max_dist` as 'tolerance points'.
        rank (torch.Tensor): Rank of each vertex, used for `box_cutoff`.
        box_cutoff (float): Threshold for `rank` to determine valid vertices/edges.
        path (str): The file path to save the generated HTML visualization.
    """
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

    # All edges will be black, so we don't need to separate valid and non-valid for coloring
    all_lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()

    skel_poly_all_edges = pv.PolyData(vertices, lines=all_lines)

    labels = labels.squeeze(-1).detach().cpu().numpy()  # (original, )
    confidences = confidences.squeeze(-1).detach().cpu().numpy()  # (original, )
    output = output.squeeze(-1).detach().cpu().numpy()  # (original, 1)

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
    plotter.add_title("Visualization")

    confident_merge_indices = np.where(combined == 3)[0]
    confident_merge_points = vertices[confident_merge_indices]

    # Calculate center of mass
    if len(confident_merge_points) > 0:
        center_of_mass = confident_merge_points.mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0)  # Use center of all points as fallback

    plotter.subplot(0)
    if skel_poly_all_edges.n_cells > 0:
        plotter.add_mesh(skel_poly_all_edges, color='black') # All edges are now black

    color_hex = "#B40426"  # Hexadecimal code for deep red
    deep_red = pv.Color(color_hex)
    color_hex = "#869BBF"  # Hexadecimal code for slightly grayish blue
    light_blue = pv.Color(color_hex)
    color_hex = "#3B4CC0"  # Hexadecimal code for deep blue
    deep_blue = pv.Color(color_hex)

    # Define the mask for valid points for the label plot
    valid_points_mask_for_label_plot = vertex_is_valid

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

    plotter.add_legend()
    plotter.add_text("Labels", font_size=14)

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True

    plotter.subplot(1)
    if skel_poly_all_edges.n_cells > 0:
        plotter.add_mesh(skel_poly_all_edges, color='black') # All edges are now black

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
            clim=[0, 1],
            scalar_bar_args={'title': 'predictions'}  # Ensure scalar bar shows full [0,1] range
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

    plotter.add_text("Output", font_size=14)
    plotter.link_views()

    plotter.subplot(2)
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
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
    """Visualization for whole cell"""
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

    cmap = create_custom_threshold_red_colormap(threshold=threshold)

    plotter.add_points(vertices_output, scalars=output, label='predictions', cmap=cmap, point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'predictions'})
    # Doesn't work for interactive it seems
    plotter.add_text("Output", font_size=14) 
    plotter.link_views()

    plotter.subplot(2)
    plotter.add_mesh(mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()
    
    plotter.export_html(save_path)  


def visualize_segclr(vertices, edges, segclr_nodes, too_far_small_radius_indices, too_far_large_radius_indices, path):
    """"Visualization for segclr embedding mapping onto l2 nodes"""
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
    """Visualization for pca SegCLR"""
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

    plotter.link_views()

    # output
    plotter.subplot(1, 2) 
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_valid, color='red')
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_non_valid, color='black')
    output = np.append(output, [0., 1.])

    plotter.add_points(vertices_output, scalars=output, label='predictions', cmap=cmap, point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'predictions'})

    plotter.link_views()

    plotter.subplot(1, 1) 
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_valid, color='red')
    if skel_poly_valid.n_cells > 0:
        plotter.add_mesh(skel_poly_non_valid, color='black')
    output_with_threshold = np.append(output_with_threshold, [0., 1.])

    plotter.add_points(vertices_output, scalars=output_with_threshold, label='threshold', cmap=cmap, point_size=10, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':f'threshold at {threshold}'})
    plotter.link_views()

    plotter.export_html(path)

def visualize_skeleton_pres(vertices, edges, labels, confidences, root_mesh, pca_segclr_percentile, has_emb, radius, output, rank, pca_pe, box_cutoff, threshold, path):
    """Visualization for skeleton (presentaion version)"""
    pv.set_jupyter_backend('trame')

    # Prepare skeleton edges for rendering (all black)
    all_lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    skel_poly_all_edges = pv.PolyData(vertices, lines=all_lines)

    pv.start_xvfb()
    # Arrange subplots horizontally: 1 row, 2 columns
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, border=True, border_color='black')

    # Calculate center of mass for camera focal point (using all vertices for simplicity)
    center_of_mass = vertices.mean(axis=0)

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True
    pv.global_theme.multi_rendering_splitting_position = 0.5 # Ensure consistent splitting

    # --- Plot 1: Root Mesh ---
    plotter.subplot(0, 0) # First subplot (row 0, column 0)
    plotter.add_title("Root Mesh")
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()

    # --- Plot 2: Skeleton with Black Points ---
    plotter.subplot(0, 1) # Second subplot (row 0, column 1)
    plotter.add_title("Skeleton with Black Points")
    plotter.add_mesh(skel_poly_all_edges, color='black') # All edges black
    plotter.add_points(vertices, color='black', point_size=15, render_points_as_spheres=True, show_vertices=True)
    plotter.link_views()

    plotter.export_html(path)

def visualize_labels_pres(vertices, edges, labels, confidences, root_mesh, pca_segclr_percentile, has_emb, radius, output, rank, pca_pe, box_cutoff, threshold, path):
    """Visualization for labels (presentation version)"""
    pv.set_jupyter_backend('trame')

    # Prepare skeleton edges for rendering (all black)
    all_lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    skel_poly_all_edges = pv.PolyData(vertices, lines=all_lines)

    pv.start_xvfb()
    # Arrange subplots horizontally: 1 row, 2 columns
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, border=True, border_color='black')

     # Process labels and confidences for coloring
    combined = labels * 2 + confidences
    c_m = (combined == 3).astype(bool)  # Confident Merge (labels=1, confidences=1)
    nc_nm = (combined == 0).astype(bool) # Non-Confident Non-Merge (labels=0, confidences=0)
    c_nm = (combined == 1).astype(bool)  # Confident Non-Merge (labels=0, confidences=1)

    # Calculate center of mass for camera focal point
    confident_merge_indices = np.where(c_m)[0]
    if len(confident_merge_indices) > 0:
        center_of_mass = vertices[confident_merge_indices].mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0) # Fallback if no confident merges

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True
    pv.global_theme.multi_rendering_splitting_position = 0.5 # Ensure consistent splitting

    # --- Plot 1: Root Mesh ---
    plotter.subplot(0, 0) # First subplot (row 0, column 0)
    plotter.add_title("Root Mesh")
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()


    # --- Plot 2: Labels ---
    plotter.subplot(0, 1)
    plotter.add_title("Labels")
    plotter.add_mesh(skel_poly_all_edges, color='black') # All edges black

    # Define colors for labels using pv.Color for consistency and PyVista compatibility
    deep_red = pv.Color("#B40426") # Confident Merge Error
    light_blue = pv.Color("#869BBF") # Non-Confident Non-Merge Error
    deep_blue = pv.Color("#3B4CC0") # Confident Non-Merge Error
    
    vertex_is_valid = rank < box_cutoff

    # Define the mask for valid points for the label plot
    valid_points_mask_for_label_plot = vertex_is_valid

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
    plotter.link_views()

    plotter.export_html(path)

def visualize_segclr_pres(vertices, edges, labels, confidences, root_mesh, pca_segclr_percentile, has_emb, radius, output, rank, pca_pe, box_cutoff, threshold, path):
    """Visualization for pca SegCLR (presentation version)"""
    pv.set_jupyter_backend('trame')

    # Prepare skeleton edges for rendering (all black)
    all_lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    skel_poly_all_edges = pv.PolyData(vertices, lines=all_lines)

    pv.start_xvfb()
    # Arrange subplots horizontally
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, border=True, border_color='black')

    # Determine valid vertices based on box_cutoff
    vertex_is_valid = rank < box_cutoff

    # Process labels and confidences for coloring
    combined = labels * 2 + confidences
    c_m = (combined == 3).astype(bool)  # Confident Merge (labels=1, confidences=1)
    nc_nm = (combined == 0).astype(bool) # Non-Confident Non-Merge (labels=0, confidences=0)
    c_nm = (combined == 1).astype(bool)  # Confident Non-Merge (labels=0, confidences=1)

    # Calculate center of mass for camera focal point
    confident_merge_indices = np.where(c_m)[0]
    if len(confident_merge_indices) > 0:
        center_of_mass = vertices[confident_merge_indices].mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0) # Fallback if no confident merges

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True
    pv.global_theme.multi_rendering_splitting_position = 0.5 # Ensure consistent splitting

    # --- Plot 1: Root Mesh ---
    plotter.subplot(0, 0) # First subplot (0-indexed)
    plotter.add_title("Root Mesh")
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()

    # --- Plot 2: PCA Percentile ---
    plotter.subplot(0, 2)
    plotter.add_mesh(skel_poly_all_edges, color='black')
    pca_segclr_percentile[has_emb == False] = np.ones(3)
    plotter.add_points(vertices, scalars=pca_segclr_percentile, point_size=10, rgb=True, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'segclr_pca'})
    plotter.link_views()


    # --- Plot 3: Labels ---
    plotter.subplot(0, 1)
    plotter.add_title("Labels")
    plotter.add_mesh(skel_poly_all_edges, color='black') # All edges black

    # Define colors for labels using pv.Color for consistency and PyVista compatibility
    deep_red = pv.Color("#B40426") # Confident Merge Error
    light_blue = pv.Color("#869BBF") # Non-Confident Non-Merge Error
    deep_blue = pv.Color("#3B4CC0") # Confident Non-Merge Error

    # Points for coloring based on labels/confidence, only for valid vertices
    if len(vertices[c_m]) > 0:
        plotter.add_points(vertices[c_m], color=deep_red, label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[nc_nm]) > 0:
        plotter.add_points(vertices[nc_nm], color=light_blue, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[c_nm]) > 0:
        plotter.add_points(vertices[c_nm], color=deep_blue, label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    plotter.link_views()

    plotter.export_html(path)

def visualize_map_pres(vertices, edges, labels, confidences, root_mesh, pca_segclr_percentile, has_emb, radius, output, rank, pca_pe, box_cutoff, threshold, path):
    """Visualization for pca positional encoding (presentation version)"""
    pv.set_jupyter_backend('trame')

    # Prepare skeleton edges for rendering (all black)
    all_lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    skel_poly_all_edges = pv.PolyData(vertices, lines=all_lines)

    pv.start_xvfb()
    # Arrange subplots horizontally
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, border=True, border_color='black')

    # Determine valid vertices based on box_cutoff
    vertex_is_valid = rank < box_cutoff

    # Process labels and confidences for coloring
    combined = labels * 2 + confidences
    c_m = (combined == 3).astype(bool)  # Confident Merge (labels=1, confidences=1)
    nc_nm = (combined == 0).astype(bool) # Non-Confident Non-Merge (labels=0, confidences=0)
    c_nm = (combined == 1).astype(bool)  # Confident Non-Merge (labels=0, confidences=1)

    # Calculate center of mass for camera focal point
    confident_merge_indices = np.where(c_m)[0]
    if len(confident_merge_indices) > 0:
        center_of_mass = vertices[confident_merge_indices].mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0) # Fallback if no confident merges

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True
    pv.global_theme.multi_rendering_splitting_position = 0.5 # Ensure consistent splitting

    # --- Plot 1: Root Mesh ---
    plotter.subplot(0, 0) # First subplot (0-indexed)
    plotter.add_title("Root Mesh")
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()

    # --- Plot 2: Map PE ---
    plotter.subplot(0, 2)
    plotter.add_mesh(skel_poly_all_edges, color='black')
    plotter.add_points(vertices, scalars=pca_pe, point_size=10, rgb=True, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'pe_pca'})
    plotter.link_views()


    # --- Plot 4: Labels ---
    plotter.subplot(0, 1)
    plotter.add_title("Labels")
    plotter.add_mesh(skel_poly_all_edges, color='black') # All edges black

    # Define colors for labels using pv.Color for consistency and PyVista compatibility
    deep_red = pv.Color("#B40426") # Confident Merge Error
    light_blue = pv.Color("#869BBF") # Non-Confident Non-Merge Error
    deep_blue = pv.Color("#3B4CC0") # Confident Non-Merge Error

    # Points for coloring based on labels/confidence, only for valid vertices
    if len(vertices[c_m]) > 0:
        plotter.add_points(vertices[c_m], color=deep_red, label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[nc_nm]) > 0:
        plotter.add_points(vertices[nc_nm], color=light_blue, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[c_nm]) > 0:
        plotter.add_points(vertices[c_nm], color=deep_blue, label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    plotter.link_views()

    plotter.export_html(path)

def visualize_map_segclr_pres(vertices, edges, labels, confidences, root_mesh, pca_segclr_percentile, has_emb, radius, output, rank, pca_pe, box_cutoff, threshold, path):
    """Visualization for pca SegCLR and pca positional encoding (presenation version)"""
    pv.set_jupyter_backend('trame')

    # Prepare skeleton edges for rendering (all black)
    all_lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    skel_poly_all_edges = pv.PolyData(vertices, lines=all_lines)

    pv.start_xvfb()
    # Arrange subplots horizontally
    plotter = pv.Plotter(shape=(1, 4), off_screen=True, border=True, border_color='black')

    # Determine valid vertices based on box_cutoff
    vertex_is_valid = rank < box_cutoff

    # Process labels and confidences for coloring
    combined = labels * 2 + confidences
    c_m = (combined == 3).astype(bool)  # Confident Merge (labels=1, confidences=1)
    nc_nm = (combined == 0).astype(bool) # Non-Confident Non-Merge (labels=0, confidences=0)
    c_nm = (combined == 1).astype(bool)  # Confident Non-Merge (labels=0, confidences=1)

    # Calculate center of mass for camera focal point
    confident_merge_indices = np.where(c_m)[0]
    if len(confident_merge_indices) > 0:
        center_of_mass = vertices[confident_merge_indices].mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0) # Fallback if no confident merges

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True
    pv.global_theme.multi_rendering_splitting_position = 0.5 # Ensure consistent splitting

    # --- Plot 1: Root Mesh ---
    plotter.subplot(0, 0) # First subplot (0-indexed)
    plotter.add_title("Root Mesh")
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()

    # --- Plot 3: PCA Percentile ---
    plotter.subplot(0, 2)
    plotter.add_mesh(skel_poly_all_edges, color='black')
    pca_segclr_percentile[has_emb == False] = np.ones(3)
    plotter.add_points(vertices, scalars=pca_segclr_percentile, point_size=15, rgb=True, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'segclr_pca'})
    plotter.link_views()

    # --- Plot 2: Map PE ---
    plotter.subplot(0, 3)
    plotter.add_mesh(skel_poly_all_edges, color='black')
    plotter.add_points(vertices, scalars=pca_pe, point_size=15, rgb=True, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'pe_pca'})
    plotter.link_views()

    # --- Plot 4: Labels ---
    plotter.subplot(0, 1)
    plotter.add_title("Labels")
    plotter.add_mesh(skel_poly_all_edges, color='black') # All edges black

    # Define colors for labels using pv.Color for consistency and PyVista compatibility
    deep_red = pv.Color("#B40426") # Confident Merge Error
    light_blue = pv.Color("#869BBF") # Non-Confident Non-Merge Error
    deep_blue = pv.Color("#3B4CC0") # Confident Non-Merge Error

    # Points for coloring based on labels/confidence, only for valid vertices
    if len(vertices[c_m]) > 0:
        plotter.add_points(vertices[c_m], color=deep_red, label='confident merge error', point_size=15, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[nc_nm]) > 0:
        plotter.add_points(vertices[nc_nm], color=light_blue, label='non-confident non-merge error', point_size=15, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[c_nm]) > 0:
        plotter.add_points(vertices[c_nm], color=deep_blue, label='confident non-merge error', point_size=15, render_points_as_spheres=True, show_vertices=True)
    plotter.link_views()

    plotter.export_html(path)

def visualize_map_segclr_output_pres(vertices, edges, labels, confidences, root_mesh, pca_segclr_percentile, has_emb, radius, output, rank, pca_pe, box_cutoff, threshold, path):
    """Visualization for pca SegCLR, positional encoding and predictions (presentation version)"""
    pv.set_jupyter_backend('trame')

    # Prepare skeleton edges for rendering (all black)
    all_lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    skel_poly_all_edges = pv.PolyData(vertices, lines=all_lines)

    pv.start_xvfb()
    # Arrange subplots horizontally
    plotter = pv.Plotter(shape=(1, 5), off_screen=True, border=True, border_color='black')

    # Determine valid vertices based on box_cutoff
    vertex_is_valid = rank < box_cutoff

    # Process labels and confidences for coloring
    combined = labels * 2 + confidences
    c_m = (combined == 3).astype(bool)  # Confident Merge (labels=1, confidences=1)
    nc_nm = (combined == 0).astype(bool) # Non-Confident Non-Merge (labels=0, confidences=0)
    c_nm = (combined == 1).astype(bool)  # Confident Non-Merge (labels=0, confidences=1)

    # Calculate center of mass for camera focal point
    confident_merge_indices = np.where(c_m)[0]
    if len(confident_merge_indices) > 0:
        center_of_mass = vertices[confident_merge_indices].mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0) # Fallback if no confident merges

    plotter.camera.tight()
    plotter.camera.focal_point = center_of_mass
    plotter.subplot_border_visible = True
    pv.global_theme.multi_rendering_splitting_position = 0.5 # Ensure consistent splitting

    # --- Plot 1: Root Mesh ---
    plotter.subplot(0, 0) # First subplot (0-indexed)
    plotter.add_title("Root Mesh")
    plotter.add_mesh(root_mesh, color="lightgrey", opacity=0.5)
    plotter.link_views()

    # --- Plot 3: PCA Percentile ---
    plotter.subplot(0, 1)
    plotter.add_mesh(skel_poly_all_edges, color='black')
    pca_segclr_percentile[has_emb == False] = np.ones(3)
    plotter.add_points(vertices, scalars=pca_segclr_percentile, point_size=10, rgb=True, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'segclr_pca'})
    plotter.link_views()

    # --- Plot 2: Map PE ---
    plotter.subplot(0, 2)
    plotter.add_mesh(skel_poly_all_edges, color='black')
    plotter.add_points(vertices, scalars=pca_pe, point_size=10, rgb=True, render_points_as_spheres=True, show_vertices=True, scalar_bar_args={'title':'pe_pca'})
    plotter.link_views()

    # --- Plot 4: Labels ---
    plotter.subplot(0, 3)
    plotter.add_title("Labels")
    plotter.add_mesh(skel_poly_all_edges, color='black') # All edges black

    # Define colors for labels using pv.Color for consistency and PyVista compatibility
    deep_red = pv.Color("#B40426") # Confident Merge Error
    light_blue = pv.Color("#869BBF") # Non-Confident Non-Merge Error
    deep_blue = pv.Color("#3B4CC0") # Confident Non-Merge Error

    # Points for coloring based on labels/confidence, only for valid vertices
    if len(vertices[c_m]) > 0:
        plotter.add_points(vertices[c_m], color=deep_red, label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[nc_nm]) > 0:
        plotter.add_points(vertices[nc_nm], color=light_blue, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[c_nm]) > 0:
        plotter.add_points(vertices[c_nm], color=deep_blue, label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    plotter.link_views()

    # --- Plot 5: Output ---
    plotter.subplot(0, 4)
    plotter.add_title("Output Predictions")
    plotter.add_mesh(skel_poly_all_edges, color='black') # All edges black

    # Prepare scalars for the output plot, including dummy values for 0 and 1 range for scalar bar
    output_scalars_for_colorbar = np.append(output, [0., 1.])
    vertices_for_output_colorbar = np.concatenate((vertices, np.zeros((1, 3)), np.ones((1, 3))), axis=0)

    # Use the custom threshold colormap for output
    cmap_custom_threshold = create_custom_threshold_red_colormap(threshold=threshold)

    # Add valid points with the colormap based on their 'output' scalar values
    valid_vertices_output_plot = vertices[vertex_is_valid]
    valid_output_scalars = output[vertex_is_valid]
    if len(valid_vertices_output_plot) > 0:
        plotter.add_points(
            valid_vertices_output_plot,
            scalars=valid_output_scalars,
            label='Predictions',
            cmap=cmap_custom_threshold,
            point_size=10,
            render_points_as_spheres=True,
            show_vertices=True,
            scalar_bar_args={'title': 'Predictions'}
        )

    # Add invalid points (rank >= box_cutoff) in black for the Output plot
    invalid_vertices_for_output_plot = vertices[~vertex_is_valid]
    if len(invalid_vertices_for_output_plot) > 0:
        plotter.add_points(
            invalid_vertices_for_output_plot,
            color='black',
            label='Invalid Points (rank >= cutoff)',
            point_size=5,
            render_points_as_spheres=True,
            show_vertices=True
        )
    plotter.link_views()

    plotter.export_html(path)
if __name__ == "__main__":
    ckpt_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/"   
    run_id = 'AUT-330' # Baseline
    epoch = 60
    # run_id = 'AUT-331' # No SegCLR
    # epoch = 40
    
    run_dir = f'{ckpt_dir}{run_id}/'
    with open(f'{run_dir}config.json', 'r') as f:
        config = json.load(f)

    config['data']["data_dir"] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/"
    config['data']['all_roots'] = "all_roots.txt"
    config['data']['train_roots'] = "train_roots.txt"
    config['data']['val_roots'] = "val_roots.txt"
    config['data']['test_roots'] = "test_roots.txt"

    if run_id == 'AUT-330' or run_id == 'AUT-331':
        config['data']['labels_dir'] = "labels_at_1300_ignore_inbetween/"

    config['loader']['fov'] = 250
    data = AutoProofDataset(config, 'all')
    model = create_model(config)
    ckpt_path = f'{run_dir}model_{epoch}.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    max_dist = config['trainer']['max_dist']
    config['trainer']['visualize_cutoff'] = 4000
    config['trainer']['branch_degrees'] = [3, 4, 5]
    box_cutoff = config['data']['box_cutoff']

    # Example root
    roots = ['864691135864976604_000']

    for root in roots:
        num_initial_vertices = data.get_num_initial_vertices(root)
        print("num_initial_vertice", num_initial_vertices, "for root", root)
        if num_initial_vertices < config['trainer']['visualize_cutoff']:
            print("getting root output")
            vertices, edges, labels, confidences, output, root_mesh, is_proofread, num_initial_vertices, dist_to_error, segclr_emb, has_segclr_emb, radius, rank, pe = get_root_output(model, device, data, root)
            print("is_proofread", is_proofread)
            
            # visual basic
            path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/test/{root}_train_vis.html'
            print("visualizing root")
            visualize(vertices, edges, labels, confidences, output, root_mesh, dist_to_error, max_dist, config['trainer']['show_tol'], rank, box_cutoff, path)
            
            # pres visual basic
            path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/test/{root}_pres_vis.html'
            print("visualizing pres root")
            visualize_pres(vertices, edges, labels, confidences, output, root_mesh, dist_to_error, max_dist, config['trainer']['show_tol'], rank, box_cutoff, path)

            # Everything below is for visuals with multiple plots including pca for pe and segclr

            # visualize segclr
            # path = ''
            # visualize_segclr(vertices, edges, path)
        
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
            # percentile_95 = np.percentile(pe_rgb, 95, axis=0)
            # percentile_5 = np.percentile(pe_rgb, 5, axis=0)
            
            # range_vals = percentile_95 - percentile_5
            # range_vals[range_vals == 0] = 1.0
            # pe_scaled_percentile = (pe_rgb - percentile_5) / range_vals
            # pe_clipped_percentile = np.clip(pe_scaled_percentile, 0, 1)

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
            
            # range_vals = percentile_95 - percentile_5
            # range_vals[range_vals == 0] = 1.0
            # segclr_scaled_percentile = (segclr_rgb - percentile_5) / range_vals
            # segclr_clipped_percentile = np.clip(segclr_scaled_percentile, 0, 1)

            # threshold = 0.05
            # output_with_threshold = np.where(output > threshold, 1., 0.)
            # print("visualizing inputs")

            # path = ''
            # visualize_labels_pres(vertices, edges, labels, confidences, root_mesh, segclr_clipped_percentile, has_segclr_emb, radius, output, rank, pe_clipped_percentile, box_cutoff, threshold, path)

            # path = ''
            # visualize_skeleton_pres(vertices, edges, labels, confidences, root_mesh, segclr_clipped_percentile, has_segclr_emb, radius, output, rank, pe_clipped_percentile, box_cutoff, threshold, path)

            # path = ''
            # visualize_segclr_pres(vertices, edges, labels, confidences, root_mesh, segclr_clipped_percentile, has_segclr_emb, radius, output, rank, pe_clipped_percentile, box_cutoff, threshold, path)
            # path = ''
            # visualize_map_pres(vertices, edges, labels, confidences, root_mesh, segclr_clipped_percentile, has_segclr_emb, radius, output, rank, pe_clipped_percentile, box_cutoff, threshold, path)

            # path = ''
            # visualize_map_segclr_pres(vertices, edges, labels, confidences, root_mesh, segclr_clipped_percentile, has_segclr_emb, radius, output, rank, pe_clipped_percentile, box_cutoff, threshold, path)

            # path = ''
            # visualize_map_segclr_output_pres(vertices, edges, labels, confidences, root_mesh, segclr_clipped_percentile, has_segclr_emb, radius, output, rank, pe_clipped_percentile, box_cutoff, threshold, path)