from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list_torch_skip_diag, prune_edges
from auto_proof.code.pre import data_utils
from auto_proof.code.model import create_model
from auto_proof.code.utils import get_root_output

import torch
import torch.nn as nn
import json
import numpy as np
from tqdm import tqdm
from torch.multiprocessing import Manager
import networkx as nx
import h5py
from torch.utils.data import Dataset, DataLoader, random_split

class ObjectDetectionDataset(Dataset):
    def __init__(self, config, root_to_output, is_confident):
        self.config = config

        self.seed_index = config['loader']['seed_index']
        self.fov = config['loader']['fov']
        self.features_dir = config['data']['features_dir']
        self.thresholds = config['trainer']['thresholds']
        self.is_confident = is_confident
        self.max_cloud = config['trainer']['max_cloud']

        self.roots = list(root_to_output.keys())
        self.manager = Manager()
        self.root_to_output = self.manager.dict(root_to_output)
        self.threshold_to_metrics = self.manager.dict()
        for threshold in self.thresholds:
            self.threshold_to_metrics[threshold] = self.manager.dict({'label_tp': 0, 'output_tp': 0, 'fn': 0, 'fp': 0})

    def __len__(self):
        return len(self.roots)

    def __getmetrics__(self):
        return dict(self.threshold_to_metrics)

    def __getitem__(self, index):
        root = self.roots[index]
        output = self.root_to_output[root]
        data_path = f'{self.features_dir}{root}.hdf5'
        try:
            with h5py.File(data_path, 'r') as f:
                labels = f['label'][:]
                confidence = f['confidence'][:]

                # Not adding rank as a feature
                rank_num = f'rank_{self.seed_index}'
                rank = f[rank_num][:]

                edges = f['edges'][:]

                if len(labels) > self.fov:
                    indices = np.where(rank < self.fov)[0]
                    labels = labels[indices]
                    confidence = confidence[indices]
                    edges = prune_edges(edges, indices)

                if is_confident:
                    conf_indices = np.where(confidence == 1)[0]
                    output = output[conf_indices]
                    labels = labels[conf_indices]
                    edges = prune_edges(edges, conf_indices)

                threshold_to_output = {}
                for threshold in self.thresholds:
                    threshold_to_output[threshold] = np.where(output < threshold, 0, 1)
                 
                node_attributes = {}
                for i in range(len(labels)):
                    node_attributes[i] = {'label': labels[i]}
                    for threshold in self.thresholds:
                        node_attributes[i][str(threshold)] = threshold_to_output[threshold][i]

                g = nx.Graph()
                g.add_edges_from(edges)
                g.add_nodes_from([(i, node_attributes[i]) for i in range(len(labels))])

                print("first node", g.nodes[0])

                print("original label error indices", np.where(labels == 0)[0])

                label_components = connected_components_by_attribute(g, 'label')
                label_components = [s for s in label_components if len(s) <= self.max_cloud]

                for component in label_components:
                    print("label component", component)

                for threshold in self.thresholds:
                    print("treshold:", threshold)

                    output_components = connected_components_by_attribute(g, str(threshold))
                    output_components = [s for s in output_components if len(s) <= self.max_cloud]
                
                    # for component in output_components:
                    #     print("output components", component)

                    label_tp, fn = count_shared_and_unshared(label_components, output_components)
                    output_tp, fp = count_shared_and_unshared(output_components, label_components)
                    print("label_tp", label_tp)
                    print("output_tp", output_tp)
                    print("fn", fn)
                    print("fp", fp)
                    self.threshold_to_metrics[threshold]['label_tp'] += label_tp
                    self.threshold_to_metrics[threshold]['output_tp'] += output_tp
                    self.threshold_to_metrics[threshold]['fn'] += fn
                    self.threshold_to_metrics[threshold]['fp'] += fp

        except Exception as e:
            print("root: ", root, "error: ", e)
            return None
            
        return 1

# def build_dataloader(config, dataset, run, mode):
#     num_workers = config['loader']['num_workers']
#     batch_size = config['loader']['batch_size']
#     shuffle = False

#     if mode == 'root' or mode == 'train':
#         shuffle = True

#     prefetch_factor = config['loader']['prefetch_factor']
#     if prefetch_factor == 0:
#         prefetch_factor = None
    # Might need to not pin memory here
#     return DataLoader(
#             dataset=dataset,
#             batch_size=batch_size, 
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=True,
#             persistent_workers=True,
#             prefetch_factor=prefetch_factor)

def object_detection(config, root, edges, labels, confidence, output, thresholds, is_confident):
    edges = edges.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().squeeze(-1)
    confidence = confidence.detach().cpu().numpy().squeeze(-1)
    output = output.detach().cpu().numpy().squeeze(-1)

    # We would also need to prune here normally based off of the fov
    # Normally we would be given a map from root to masked output or conf output

    if is_confident:
        conf_indices = np.where(confidence == 1)[0]
        output = output[conf_indices]
        labels = labels[conf_indices]
        edges = prune_edges(edges, conf_indices)

    threshold_to_output = {}
    for threshold in thresholds:
        threshold_to_output[threshold] = np.where(output < threshold, 0, 1)
    
    node_attributes = {}
    for i in range(len(labels)):
        node_attributes[i] = {'label': labels[i]}
        for threshold in thresholds:
            node_attributes[i][str(threshold)] = threshold_to_output[threshold][i]

    max_cloud = config['trainer']['max_cloud']
    # max_cloud = 6

    g = nx.Graph()
    g.add_edges_from(edges)
    # g.add_nodes_from([(node_index, {'label': labels[node_index], 'output': output[node_index]}) for node_index in range(len(labels))])
    g.add_nodes_from([(i, node_attributes[i]) for i in range(len(labels))])

    print("first node", g.nodes[0])

    print("original label error indices", np.where(labels == 0)[0])

    label_components = connected_components_by_attribute(g, 'label')
    label_components = [s for s in label_components if len(s) <= max_cloud]

    for component in label_components:
        print("label component", component)

    for threshold in thresholds:
        print("treshold:", threshold)

        output_components = connected_components_by_attribute(g, str(threshold))
        output_components = [s for s in output_components if len(s) <= max_cloud]
    
        # for component in output_components:
        #     print("output components", component)

        label_tp, fn = count_shared_and_unshared(label_components, output_components)
        output_tp, fp = count_shared_and_unshared(output_components, label_components)
        print("label_tp", label_tp)
        print("output_tp", output_tp)
        print("fn", fn)
        print("fp", fp)

def count_shared_and_unshared(list_of_sets1, list_of_sets2):
    shared = 0
    unshared = 0
    for curr_set1 in list_of_sets1:
        not_shared = True
        for curr_set2 in list_of_sets2:
            if curr_set1.intersection(curr_set2):
                not_shared = False
                shared += 1
                break
        
        if not_shared:
            unshared += 1
    return shared, unshared

def connected_components_by_attribute(graph, attribute):
    value = 0 # Represents error
    subgraph_nodes = [node for node, data in graph.nodes(data=True) if data.get(attribute) == value]
    subgraph = graph.subgraph(subgraph_nodes)
    components = list(nx.connected_components(subgraph))
    return components

if __name__ == "__main__":
    ckpt_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/"   
    run_id = 'AUT-215'
    run_dir = f'{ckpt_dir}{run_id}/'
    ckpt_path = f'{run_dir}model_55.pt'
    with open(f'{run_dir}config.json', 'r') as f:
        config = json.load(f)

    data = AutoProofDataset(config, 'root')
    
    model = create_model(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # roots = [864691135463333789]
    # roots = [864691136379030485]
    # roots = [864691136443843459]
    roots = [864691136443843459, 864691135463333789]
    config['trainer']['thresholds'] = [0.5, 0.8]
    config['trainer']['max_cloud'] = 15

    root_to_output = {}

    for root in roots:
        print("Getting root output")
        # Remember this doesn't pre mask so only use things above fov for testing right now
        vertices, edges, labels, confidence, output, _, _, _, _ = get_root_output(model, device, data, root)

        output = output.detach().cpu().numpy().squeeze(-1)
        # print("Starting object detection")
        # object_detection(config, root, edges, labels, confidence, output, thresholds, False)

        root_to_output[root] = output

    config['loader']['batch_size'] = 1
    config['loader']['num_workers'] = 1
    is_confident = False
    print("creating dataset")
    obj_det_data = ObjectDetectionDataset(config, root_to_output, is_confident)

    print("getting item")
    obj_det_data.__getitem__(0)
    obj_det_data.__getitem__(1)
    print("metrics at threshold 0.5", obj_det_data.__getmetrics__()[0.5])
    print("metrics at threshold 0.8", obj_det_data.__getmetrics__()[0.8])