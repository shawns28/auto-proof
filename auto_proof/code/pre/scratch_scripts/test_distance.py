from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list_numpy_skip_diag, adjency_to_edge_list_torch_skip_diag 
from auto_proof.code.pre import data_utils
from auto_proof.code.model import create_model

import torch
import torch.nn as nn
import json
import pyvista as pv
import numpy as np
from tqdm import tqdm
import networkx as nx
from collections import deque
import torch.multiprocessing as mp

CONFIG_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/base_config.json'

# Not relevant anymore
def test_distance(config, model, device, data, root, metrics):
    print("entering the method")
    # with lock:
    # metrics = {}
    # thresholds = [0.5]
    # for threshold in thresholds:
    #     metrics[threshold] = {'tp': 0, 'fp': 0, 'fn': 0}
    model.eval()
    with torch.no_grad():
        idx = data.get_root_index(root)
        sample = data.__getitem__(idx)
        input, labels, confidence, adj = sample 
        input = input.float().to(device).unsqueeze(0) # (1, fov, d)
        labels = labels.float().to(device) # (fov, 1)
        confidence = confidence.float().to(device) # (fov, 1)
        adj = adj.float().to(device).unsqueeze(0) # (1, fov, fov)

        output = model(input, adj) # (1, fov, 1)

        output = output.squeeze(0) # (fov, 1)
        # print("output after squeeze", output)

        # original represents the min of original points that weren't buffered and fov
        mask = labels != -1
        mask = mask.squeeze(-1) # (original)
        
        # Apply mask to get original points
        output = output[mask] # (original, 2)
        input = input.squeeze(0)[mask] # (original, d)
        labels = labels[mask] # (original, 1)
        labels = labels.squeeze(-1) # (original)
        confidence = confidence[mask] # (original, 1)
        confidence.squeeze(-1) # (original)
        adj = adj.squeeze(0)
        # adj = adj.squeeze(0)[mask, :][:, mask] # (original, original) # Don't think I need this
        # print("adj", adj)
    
        edges = adjency_to_edge_list_torch_skip_diag(adj)
        # print("edges", edges)

        # print("output", output)
        output_merge_prob = output[..., 0] # (original)
        # print("output_merge_prob", output_merge_prob)

        edges = edges.detach().cpu().numpy()
        output_merge_prob = output_merge_prob.detach().cpu().numpy()
        # Inverting since merge errors should be 1 for precision/recall
        labels = 1 - labels.detach().cpu().numpy()
        # print("labels", labels)

        g = create_graph(edges)
        max_dist = config['trainer']['max_dist']
       
        for threshold in metrics.keys():
            output_converted = np.where(output_merge_prob >= threshold, 1, 0)
            # print("output converted", output_converted)
            tp = np.sum((output_converted == 1) & (labels == 1))
            # metrics[threshold]['tp'] += tp
            # print("tp", tp)
            fp_ind = np.where((output_converted == 1) & (labels == 0))[0]
            # print("fp arr", fp_ind)
            fn_ind = np.where((output_converted == 0) & (labels == 1))[0]
            # print("fn_arr", fn_ind)
            
            fp = np.sum((output_converted == 1) & (labels == 0))
            # print("fp", fp)
            fn = np.sum((output_converted == 0) & (labels == 1))
            # print("fn", fn)

            curr_tp, fp, fn = process_mismatch(g, fp_ind, fn_ind, max_dist, labels, output_converted, metrics, threshold)
            tp += curr_tp
            metrics[threshold]['tp'] += tp
            metrics[threshold]['fp'] += fp
            metrics[threshold]['fn'] += fn
            # print("tp after", metrics[threshold]['tp'])
            # print("fp after", metrics[threshold]['fp'])
            # print("fn after", metrics[threshold]['fn'])           
    

# Create a graph with the vertices having the index and label maybe
# For each threshold pass in the indices of fp and fn
# Each indice goes to its closest node and checks the distance and associated label
# If it finds one within distance count it as a tp/tn and if not count it as fp/fn

# Keep track of the different tp/pn fp/fn in a map for each threshold

def create_graph(edges):
    g = nx.Graph()
    g.add_edges_from(edges)
    return g

def process_mismatch(g, fp_ind, fn_ind, max_dist, labels, output_converted, metrics, threshold):
    #TODO: Need to also compare this to using bfs edges and searching through the list

    tp = 0
    fp = 0
    fn = 0

    # target_label is 1 because we want to find closest labeled error
    for ind in fp_ind:
        res = bfs_with_max_dist(g, ind, max_dist, labels, output_converted, fp_condition)
        if not res:
            fp += 1
    # print("switching to fn")
    for ind in fn_ind:
        res = bfs_with_max_dist(g, ind, max_dist, labels, output_converted, fn_condition)
        if res:
            tp += 1
        else:
            fn += 1
    return tp, fp, fn

# If max_dist is none then it looks through everything
# If max_dist is 0 then it returns false
def bfs_with_max_dist(g, start_node, max_dist, labels, output_converted, target_condition):
    if max_dist == 0:
        return False
    
    visited = {start_node}
    q = deque()
    q.append((start_node, 0))

    while q:
        curr_node, curr_dist = q.popleft()
        # print("curr node processsed", curr_node)
        # print("curr dist", curr_dist)
        # First node that enters can't be this due to definition of fp anyway
        if target_condition(curr_node, labels, output_converted) and curr_node != start_node:
            # print("found successful condition", curr_node)
            return True
        
        if max_dist is not None and curr_dist >= max_dist:
            continue

        for neighbor in g.neighbors(curr_node):
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, curr_dist + 1))

    return False

def fp_condition(node, labels, output_converted):
    return labels[node] == 1

def fn_condition(node, labels, output_converted):
    return labels[node] == 1 and output_converted[node] == 1

if __name__ == "__main__":    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    # config['model']['depth'] = 3
    # config['model']['n_head'] = 4
    config['model']['depth'] = 7
    config['model']['n_head'] = 8

    data = AutoProofDataset(config, 'test')

    model = create_model(config)
    # ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20250215_212005/model_34.pt'
    ckpt_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/20250306_124001/model_48.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load(ckpt_path))
    
    # Setting this or not still makes it go sequentially
    # mp.set_start_method('spawn', force=True)
    
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.share_memory()

    # root = 864691135937424949
    # root = 864691135778235581
    # root = 864691135463333789
    # root = 864691136443843459
    roots = [864691136443843459, 864691134918370314, 864691136903430322, 864691135382615514]

    thresholds = [0.5]
    with mp.Manager() as manager:
        metrics = manager.dict()
        # lock = manager.Lock()
        for threshold in thresholds:
            metrics[threshold] = manager.dict({'tp': 0, 'fp': 0, 'fn': 0})

        processes = []
        num_processes = 4
        for rank in range(num_processes):
            p = mp.Process(target=test_distance, args=(config, model, device, data, roots[rank], metrics))
            # p = mp.Process(target=test_distance, args=(config, model, device, data, roots[rank], metrics, lock))
            # p = mp.Process(target=test_distance, args=(config, model, device, data, roots[rank]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print("done")
        for threshold in metrics:
            print("Final tp", metrics[threshold]['tp'])
            print("Final fp", metrics[threshold]['fp'])
            print("Final fn", metrics[threshold]['fn'])
    # test_distance(model, device, data, roots[0], metrics)
