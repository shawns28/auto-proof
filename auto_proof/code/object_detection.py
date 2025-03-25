from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list_torch_skip_diag, prune_edges, build_dataloader
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
import time
import matplotlib.pyplot as plt

class ObjectDetectionDataset(Dataset):
    def __init__(self, config, root_to_output):
        self.config = config

        self.seed_index = config['loader']['seed_index']
        self.fov = config['loader']['fov']
        self.features_dir = config['data']['features_dir']
        self.thresholds = config['trainer']['thresholds']
        self.max_cloud = config['trainer']['max_cloud']

        self.roots = list(root_to_output.keys())
        self.manager = Manager()
        self.root_to_output = self.manager.dict(root_to_output)
        self.threshold_to_metrics = self.manager.dict()
        for threshold in self.thresholds:
            self.threshold_to_metrics[threshold] = self.manager.dict({
                'all_label_tp': 0,
                'all_output_tp': 0, 
                'all_fn': 0, 
                'all_fp': 0, 
                'conf_label_tp': 0,
                'conf_output_tp': 0, 
                'conf_fn': 0, 
                'conf_fp': 0, 
            })

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

                rank_num = f'rank_{self.seed_index}'
                rank = f[rank_num][:]

                edges = f['edges'][:]

                if len(labels) > self.fov:
                    indices = np.where(rank < self.fov)[0]
                    labels = labels[indices]
                    confidence = confidence[indices]
                    edges = prune_edges(edges, indices)

                assert(len(labels) == len(output))

                threshold_to_output = {}
                for threshold in self.thresholds:
                    threshold_to_output[threshold] = np.where(output < threshold, 0, 1)
                 
                node_attributes = {}
                for i in range(len(labels)):
                    node_attributes[i] = {'label': labels[i]}
                    node_attributes[i]['confidence'] = confidence[i]
                    for threshold in self.thresholds:
                        node_attributes[i][str(threshold)] = threshold_to_output[threshold][i]

                g = nx.Graph()
                g.add_edges_from(edges)
                g.add_nodes_from([(i, node_attributes[i]) for i in range(len(labels))])

                label_components, conf_label_components = connected_components_by_attribute(g, 'label')
                label_components, label_components_removed = remove_big_clouds(label_components, self.max_cloud)
                conf_label_components, _ = remove_big_clouds(conf_label_components, self.max_cloud)
                # TODO: Better way to do this because currently its glitched
                # if len(label_components_removed) > 0:
                #     print("Root has labeled component over max cloud", root)

                for threshold in self.thresholds:

                    output_components, conf_output_components = connected_components_by_attribute(g, str(threshold))
                    output_components, output_components_removed = remove_big_clouds(output_components, self.max_cloud)
                    new_conf_output_components = []
                    for conf_output_component in conf_output_components:
                        if conf_output_component.isdisjoint(output_components_removed):
                            new_conf_output_components.append(conf_output_component)
                    conf_output_components = new_conf_output_components
                    # if len(conf_output_components) > 0 and len(output_components_removed) > 0:
                    #     conf_output_components = [[conf_output_component for conf_output_component in conf_output_components if conf_output_component.isdisjoint(output_components_removed)]]

                    all_label_tp, all_fn = count_shared_and_unshared(label_components, output_components)
                    all_output_tp, all_fp = count_shared_and_unshared(output_components, label_components)
                    conf_label_tp, conf_fn = count_shared_and_unshared(conf_label_components, conf_output_components)
                    conf_output_tp, conf_fp = count_shared_and_unshared(conf_output_components, conf_label_components)
                    self.threshold_to_metrics[threshold]['all_label_tp'] += all_label_tp
                    self.threshold_to_metrics[threshold]['all_output_tp'] += all_output_tp
                    self.threshold_to_metrics[threshold]['all_fn'] += all_fn
                    self.threshold_to_metrics[threshold]['all_fp'] += all_fp
                    self.threshold_to_metrics[threshold]['conf_label_tp'] += conf_label_tp
                    self.threshold_to_metrics[threshold]['conf_output_tp'] += conf_output_tp
                    self.threshold_to_metrics[threshold]['conf_fn'] += conf_fn
                    self.threshold_to_metrics[threshold]['conf_fp'] += conf_fp
                    if threshold == 0.5 and conf_fn > 0:
                        print("root with conf fn", root)
        except Exception as e:
            print("root: ", root, "error: ", e)
            return None
            
        return 1 

def remove_big_clouds(components, max_cloud):
    big_clouds = set()
    filtered_components = []
    for cc in components:
        if len(cc) > max_cloud:
            big_clouds.update(cc)
        else:
            filtered_components.append(cc)
    return filtered_components, big_clouds

def count_shared_and_unshared(list_of_sets1, list_of_sets2):
    shared = 0
    unshared = 0
    for curr_set1 in list_of_sets1:
        not_shared = True
        for curr_set2 in list_of_sets2:
            if not curr_set1.isdisjoint(curr_set2):
                not_shared = False
                shared += 1
                break
        
        if not_shared:
            unshared += 1
    return shared, unshared

def connected_components_by_attribute(graph, attribute):
    error_value = 0 # Represents error
    subgraph_nodes = [node for node, data in graph.nodes(data=True) if data.get(attribute) == error_value]
    subgraph = graph.subgraph(subgraph_nodes)
    components = list(nx.connected_components(subgraph))

    # Confident points only
    conf_value = 1
    conf_subgraph_nodes = [node for node, data in subgraph.nodes(data=True) if data.get('confidence') == conf_value]
    conf_subgraph = subgraph.subgraph(conf_subgraph_nodes)
    conf_components = list(nx.connected_components(conf_subgraph))

    return components, conf_components

def obj_det_dataloader(config, dataset, ratio):
    num_workers = config['loader']['num_workers']
    batch_size = config['loader']['batch_size']
    shuffle = True # Always since we're taking a sample
    pin_memory = False # cpu only so don't need it
    persistent_workers= False
    prefetch_factor = None

    sample_size = int(ratio * len(dataset))
    other_size = len(dataset) - sample_size
    sample_dataset, _ = random_split(dataset, [sample_size, other_size])

    return DataLoader(
            dataset=sample_dataset,
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor)

def obj_det_plots(metrics_dict, thresholds, epoch, save_dir):
    save_path_all_label = obj_det_plot(metrics_dict, thresholds, epoch, save_dir, 'all', 'label_tp')
    save_path_all_output = obj_det_plot(metrics_dict, thresholds, epoch, save_dir, 'all', 'output_tp')
    save_path_conf_label = obj_det_plot(metrics_dict, thresholds, epoch, save_dir, 'conf', 'label_tp')
    save_path_conf_output = obj_det_plot(metrics_dict, thresholds, epoch, save_dir, 'conf', 'output_tp')
    return [('all', 'label_tp', save_path_all_label), ('all', 'output_tp', save_path_all_output), ('conf', 'label_tp', save_path_conf_label), ('conf', 'output_tp', save_path_conf_output)]

def obj_det_plot(metrics_dict, thresholds, epoch, save_dir, mode, tp_mode):
    recalls = []
    precisions = []
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        # Using label tp instead of output tp for now
        precision, recall = get_precision_and_recall(metrics_dict[threshold][f'{mode}_{tp_mode}'], metrics_dict[threshold][f'{mode}_fn'], metrics_dict[threshold][f'{mode}_fp'])
        precisions.append(precision)
        recalls.append(recall)

    plt.figure(figsize=(8, 8))
    plt.plot(recalls, precisions, marker='.', markersize=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    if mode == 'all':
        plt.title('Merge Error Object Based Precision-Recall Curve for All Nodes Epoch {epoch}')
    else:
        plt.title('Merge Error Object Based Precision-Recall Curve for Confident Nodes Epoch {epoch}')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    for i in range(len(thresholds)):
        plt.scatter(recalls[i], precisions[i], c='red', s=30, label=f'Recall: {recalls[i]:.2f}, Precision: {precisions[i]:.2f}, Threshold: {thresholds[i]}')  # Mark with a red dot
    plt.legend()
    plt.grid(True)
    save_path = f'{save_dir}obj_precision_recall_{mode}_{tp_mode}_{epoch}.png'
    plt.savefig(save_path)
    return save_path

def get_precision_and_recall(tp, fn, fp):
    precision = 0
    if (tp + fp > 0):
        precision = tp / (tp + fp)
    recall = 0
    if (tp + fn > 0):
        recall = tp / (tp + fn)
    return precision, recall

if __name__ == "__main__":
    ckpt_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/"   
    run_id = 'AUT-215'
    run_dir = f'{ckpt_dir}{run_id}/'
    epoch = 55
    ckpt_path = f'{run_dir}model_{epoch}.pt'
    with open(f'{run_dir}config.json', 'r') as f:
        config = json.load(f)
    
    model = create_model(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # roots = [864691135463333789]
    # roots = [864691136379030485]
    # roots = [864691136443843459]
    # roots = [864691136443843459, 864691135463333789]
    config['trainer']['thresholds'] = [0.01, 0.1, 0.4, 0.5, 0.6, 0.9, 0.99]
    config['trainer']['max_cloud'] = 15

    data = AutoProofDataset(config, 'val')
    config['loader']['batch_size'] = 32
    config['loader']['num_workers'] = 32
    config['trainer']['obj_loader_ratio'] = 0.1
    data_loader = build_dataloader(config, data, 'val')

    model.eval()
    with torch.no_grad():
        root_to_output = {}

        # for root in roots:
        # print("Getting root output")
        # # Remember this doesn't pre mask so only use things above fov for testing right now
        # vertices, edges, labels, confidence, output, _, _, _, _ = get_root_output(model, device, data, root)
        # output = output.detach().cpu().numpy().squeeze(-1)
        # root_to_output[root] = output

        with tqdm(total=len(data) / config['loader']['batch_size'], desc="val") as pbar:
            for i, data in enumerate(data_loader):
                # root (b, 1) input (b, fov, d), labels (b, fov, 1), conf (b, fov, 1), adj (b, fov, fov)
                roots, input, labels, confidence, dist_to_error, adj = data
                # TODO: Only send input and adj to device and make everything later numpy to make things faster?
                input = input.float().to(device)
                labels = labels.float().to(device)

                adj = adj.float().to(device)
                output = model(input, adj) # (b, fov, 1)

                sigmoid = nn.Sigmoid()
                output = sigmoid(output).squeeze(-1) # (b, fov)
                labels = labels.squeeze(-1) # (b, fov)

                mask = labels != -1
                mask = mask.detach().cpu().numpy()

                output = output.detach().cpu().numpy()
                roots = roots.detach().cpu().numpy().astype(int)
                for i in range(len(roots)):
                    root_to_output[roots[i]] = output[i][mask[i]]

                pbar.update()
        
        print("creating obj dataset")
        obj_det_data = ObjectDetectionDataset(config, root_to_output)
        obj_det_loader = obj_det_dataloader(config, obj_det_data, config['trainer']['obj_loader_ratio'])
        with tqdm(total=int(config['trainer']['obj_loader_ratio'] * len(obj_det_data)) / config['loader']['batch_size'], desc="obj det all") as pbar:
            for i, data in enumerate(obj_det_loader):
                pbar.update()
        
        metrics_dict = obj_det_data.__getmetrics__()
        print("metrics at threshold 0.5", obj_det_data.__getmetrics__()[0.5])
        print("metrics at threshold 0.99", obj_det_data.__getmetrics__()[0.99])

        # save_dir = run_dir
        save_dir = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/'

        save_path_tuples = obj_det_plots(metrics_dict, config['trainer']['thresholds'], epoch, save_dir)
    print("done")    

