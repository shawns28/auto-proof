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
import time

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
        file_before = time.time()
        try:
            with h5py.File(data_path, 'r') as f:
                file_after = time.time()
                print("file open time", file_after - file_before)
                time_1 = time.time()
                labels = f['label'][:]
                confidence = f['confidence'][:]

                rank_num = f'rank_{self.seed_index}'
                rank = f[rank_num][:]

                edges = f['edges'][:]
                time_2 = time.time()

                if len(labels) > self.fov:
                    indices = np.where(rank < self.fov)[0]
                    labels = labels[indices]
                    confidence = confidence[indices]
                    edges = prune_edges(edges, indices)
                time_3 = time.time()

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

                time_4 = time.time()
                g = nx.Graph()
                g.add_edges_from(edges)
                g.add_nodes_from([(i, node_attributes[i]) for i in range(len(labels))])
                time_5 = time.time()

                label_components, conf_label_components = connected_components_by_attribute(g, 'label')
                time_6 = time.time()
                label_components, label_components_removed = remove_big_clouds(label_components, self.max_cloud)
                conf_label_components, _ = remove_big_clouds(conf_label_components, self.max_cloud)
                if len(label_components_removed) > 0:
                    print("Root has labeled component over max cloud", root)
                time_7 = time.time()

                for threshold in self.thresholds:

                    output_components, conf_output_components = connected_components_by_attribute(g, str(threshold))
                    output_components, output_components_removed = remove_big_clouds(output_components, self.max_cloud)
                    time_8 = time.time()
                    new_conf_output_components = []
                    for conf_output_component in conf_output_components:
                        if conf_output_component.isdisjoint(output_components_removed):
                            new_conf_output_components.append(conf_output_component)
                    conf_output_components = new_conf_output_components
                    time_9 = time.time()
                    # if len(conf_output_components) > 0 and len(output_components_removed) > 0:
                    #     conf_output_components = [[conf_output_component for conf_output_component in conf_output_components if conf_output_component.isdisjoint(output_components_removed)]]

                    all_label_tp, all_fn = count_shared_and_unshared(label_components, output_components)
                    all_output_tp, all_fp = count_shared_and_unshared(output_components, label_components)
                    time_10 = time.time()
                    conf_label_tp, conf_fn = count_shared_and_unshared(conf_label_components, conf_output_components)
                    conf_output_tp, conf_fp = count_shared_and_unshared(conf_output_components, conf_label_components)
                    time_11 = time.time()
                    self.threshold_to_metrics[threshold]['all_label_tp'] += all_label_tp
                    self.threshold_to_metrics[threshold]['all_output_tp'] += all_output_tp
                    self.threshold_to_metrics[threshold]['all_fn'] += all_fn
                    self.threshold_to_metrics[threshold]['all_fp'] += all_fp
                    self.threshold_to_metrics[threshold]['conf_label_tp'] += conf_label_tp
                    self.threshold_to_metrics[threshold]['conf_output_tp'] += conf_output_tp
                    self.threshold_to_metrics[threshold]['conf_fn'] += conf_fn
                    self.threshold_to_metrics[threshold]['conf_fp'] += conf_fp
                    time_12 = time.time()
            print("time 1", time_2 - time_1)
            print("time 2", time_3 - time_2)
            print("time 3", time_4 - time_3)
            print("time 4", time_5 - time_4)
            print("time 5", time_6 - time_5)
            print("time 6", time_7 - time_6)
            print("time 7", time_8 - time_7)
            print("time 8", time_9 - time_8)
            print("time 9", time_10 - time_9)
            print("time 10", time_11 - time_10)
            print("time 11", time_12 - time_11)
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

def obj_det_dataloader(config, dataset, mode):
    num_workers = config['loader']['num_workers']
    batch_size = config['loader']['batch_size']
    shuffle = False
    pin_memory = False # cpu only so don't need it
    persistent_workers= False

    if mode == 'root' or mode == 'train':
        shuffle = True

    prefetch_factor = config['loader']['prefetch_factor']
    if prefetch_factor == 0:
        prefetch_factor = None

    # TODO: Depending on the speed we might have to randomly split the dataset in 1/2 or 1/3

    return DataLoader(
            dataset=dataset,
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor)

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

    roots = [864691135463333789]
    # roots = [864691136379030485]
    # roots = [864691136443843459]
    # roots = [864691136443843459, 864691135463333789]
    config['trainer']['thresholds'] = [0.99]
    config['trainer']['max_cloud'] = 15

    root_to_output = {}

    for root in roots:
        print("Getting root output")
        # Remember this doesn't pre mask so only use things above fov for testing right now
        vertices, edges, labels, confidence, output, _, _, _, _ = get_root_output(model, device, data, root)
        detach_time = time.time()
        output = output.detach().cpu().numpy().squeeze(-1)
        detach_post_time = time.time()
        print("detach time", detach_post_time - detach_time)
        # print("Starting object detection")
        # object_detection(config, root, edges, labels, confidence, output, thresholds, False)

        root_to_output[root] = output

    config['loader']['batch_size'] = 1
    config['loader']['num_workers'] = 1
    print("creating dataset")
    obj_det_data = ObjectDetectionDataset(config, root_to_output)
    obj_det_loader = obj_det_dataloader(config, obj_det_data, 'val')
    before_loader = time.time()
    with tqdm(total=len(obj_det_data) / config['loader']['batch_size'], desc="obj det all") as pbar:
        for i, data in enumerate(obj_det_loader):
            pbar.update()
    after_loader = time.time()
    print("loader total time", after_loader - before_loader)
    
    # print("metrics at threshold 0.5", obj_det_data.__getmetrics__()[0.5])
    # print("metrics at threshold 0.8", obj_det_data.__getmetrics__()[0.8])
    print("metrics at threshold 0.99", obj_det_data.__getmetrics__()[0.99])