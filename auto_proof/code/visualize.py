from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list_torch_skip_diag
from auto_proof.code.pre import data_utils
from auto_proof.code.model import create_model
from auto_proof.code.utils import get_root_output

import torch
import torch.nn as nn
import json
import pyvista as pv
import numpy as np
# from cloudvolume import CloudVolume
from tqdm import tqdm
import multiprocessing
# Remember to remove morphlink package that already exists in directory

# Not detaching here because it causes weird error where it wants it to 
def visualize(vertices, edges, labels, confidence, output, root_mesh, dist_to_error, max_dist, path):
    pv.set_jupyter_backend('trame')
    vertices = vertices.detach().cpu().numpy()
    edges = edges.detach().cpu().numpy()
    lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    skel_poly = pv.PolyData(vertices, lines=lines)
    
    labels = labels.squeeze(-1).detach().cpu().numpy() # (original, )
    confidence = confidence.squeeze(-1).detach().cpu().numpy() # (original, )
    output = output.squeeze(-1).detach().cpu().numpy() # (original, 1)
    dist_to_error = dist_to_error.squeeze(-1).detach().cpu().numpy() # (original, )

    combined = labels * 2 + confidence
    c_m = (combined == 1).astype(bool)
    nc_nm = (combined == 2).astype(bool)
    c_nm = (combined == 3).astype(bool)
    # 0 = non-confident merge (This can't happen)
    # 1 = confident merge
    # 2 = non-confident non-merge
    # 3 = confident non-merge

    dist_mask_inc = np.logical_and(dist_to_error > 0, dist_to_error <= max_dist)
    dist_mask_exc = np.logical_or(dist_to_error == 0, dist_to_error > max_dist)
    
    pv.global_theme.multi_rendering_splitting_position = 0.5

    pv.start_xvfb()
    plotter = pv.Plotter(shape="1/2", off_screen=True, border=True, border_color='black')
    # All labels don't work for interactive
    plotter.add_title("Visualization")

    confident_merge_indices = np.where(combined == 1)[0] 
    confident_merge_points = vertices[confident_merge_indices] 

    # Calculate center of mass
    if len(confident_merge_points) > 0:
        center_of_mass = confident_merge_points.mean(axis=0)
    else:
        center_of_mass = vertices.mean(axis=0)  # Use center of all points as fallback
    
    plotter.subplot(0)
    plotter.add_mesh(skel_poly, color='black')
    
    if len(vertices[c_m & dist_mask_exc]) > 0:
        plotter.add_points(vertices[c_m & dist_mask_exc], color='red', label='confident merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[nc_nm & dist_mask_exc]) > 0:
        color_hex = "#9ACD32"  # Hexadecimal code for yellow-green
        yellow_green = pv.Color(color_hex) 
        plotter.add_points(vertices[nc_nm & dist_mask_exc], color=yellow_green, label='non-confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[c_nm & dist_mask_exc]) > 0:
        plotter.add_points(vertices[c_nm & dist_mask_exc], color='green', label='confident non-merge error', point_size=10, render_points_as_spheres=True, show_vertices=True)
    if len(vertices[dist_mask_inc]) > 0:
        color_hex = "#FFD580" # Hexadecimal code for light 
        light_orange = pv.Color(color_hex) 
        plotter.add_points(vertices[dist_mask_inc], color=light_orange, label='tolerance points', point_size=10, render_points_as_spheres=True, show_vertices=True)
    
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
        vertices, edges, labels, confidence, output, root_mesh, is_proofread, num_intitial_vertices, dist_to_error = get_root_output(model, device, data, root)
        print("is_proofread", is_proofread)
        print("num_intitial_vertice", num_intitial_vertices)
        visualize(vertices, edges, labels, confidence, output, root_mesh, dist_to_error, path)
    except Exception as e:
        print("Failed visualization for root id: ", root, "error: ", e)
            

if __name__ == "__main__":
    ckpt_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/"   
    # run_id = 'AUT-215'
    run_id = 'AUT-246'
    run_dir = f'{ckpt_dir}{run_id}/'
    with open(f'{run_dir}config.json', 'r') as f:
        config = json.load(f)

    data = AutoProofDataset(config, 'train')
    # config['model']['depth'] = 3
    # config['model']['n_head'] = 4
    model = create_model(config)
    # ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20241104_132019/model_29'
    # ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20250212_125759/model_1.pt'
    # ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20250215_212005/model_34.pt'
    # ckpt_path = f'{run_dir}model_55.pt'
    ckpt_path = f'{run_dir}model_1.pt'
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
    roots = [864691135463333789]
    # roots = [864691135373853640, 864691135658276738, 864691135684548279, 864691135684548023, 864691135404040430, 864691135065359940, 864691135386905985, 864691135404090350, 864691135396807713, 864691135387767681, 864691135341705649, 864691135446000274, 864691135947108705, 864691135947108193, 864691135447045460, 864691135359346776, 864691135987462147]
    # roots = [864691135684548279]
    # roots = [864691135396807713]
    # roots = [864691135989085184, 864691135683792114, 864691136483519276, 864691135571609125, 864691135876419923, 864691136310246234, 864691136009015212, 864691135697685269, 864691136266911476]
    # Roots that are sus because conf and error are similar but not exact
    # roots = [864691134940047459, 864691135411419697, 864691135724417451, 864691135472111666]
    max_dist = config['trainer']['max_dist']
    # stuff = [(model, device, data, 864691135463333789), (model, device, data, 864691135778235581), (model, device, data, 864691135463333789), (model, device, data, 864691135778235581)]
    # num_processes = len(stuff)
    # with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(stuff)) as pbar:
    #     for root in pool.imap_unordered(visualize_root, stuff):
    #         pbar.update()
    config['trainer']['visualize_cutoff'] = 4400
    config['loader']['fov'] = 250
    for i in range(len(roots)):
        roots[i] = str(roots[i]) + '_000'
           
    # for i in range(10):
    for root in roots:
        # try:
        # root = data.get_random_root()
        # while data.get_num_initial_vertices(root) > config['trainer']['visualize_cutoff']:
        #     root = data.get_random_root()
        num_initial_vertices = data.get_num_initial_vertices(root)
        print("num_initial_vertice", num_initial_vertices, "for root", root)
        if num_initial_vertices < config['trainer']['visualize_cutoff']:
            path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/visualize_{root}_ckpt55_250_fov.html'
            print("getting root output")
            vertices, edges, labels, confidence, output, root_mesh, is_proofread, num_initial_vertices, dist_to_error = get_root_output(model, device, data, root)
            print("is_proofread", is_proofread)
            visualize(vertices, edges, labels, confidence, output, root_mesh, dist_to_error, max_dist, path)
        # except Exception as e:
        #     print("Failed visualization for root id: ", root, "error: ", e)
        #     continue

    # roots = [864691136831441518, 864691136379030485, 864691135585405308, 864691135278463137, 864691136673368967, 864691136084808300, 864691135976247107, 864691135946876001, 864691135725734719, 864691135740501099, 864691135772003147, 864691136024491577]
    # for root in roots:
    #     print("root", root, "num initial", data.get_num_initial_vertices(root))