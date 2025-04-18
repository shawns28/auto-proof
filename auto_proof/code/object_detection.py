from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list_torch_skip_diag, prune_edges, build_dataloader
from auto_proof.code.pre import data_utils
from auto_proof.code.model import create_model

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

        # self.seed_index = config['loader']['seed_index']
        self.fov = config['loader']['fov']
        self.data_dir = config['data']['data_dir']
        self.features_dir = f'{self.data_dir}{config['data']['features_dir']}'
        self.labels_dir = f'{self.data_dir}{config['data']['labels_dir']}'
        self.box_cutoff = config['data']['box_cutoff']
        self.thresholds = config['trainer']['thresholds']
        self.obj_det_error_cloud_ratio = config['trainer']['obj_det_error_cloud_ratio']

        self.roots = list(root_to_output.keys())
        self.manager = Manager()
        self.root_to_output = self.manager.dict(root_to_output)
        self.threshold_to_metrics = self.manager.dict()
        for threshold in self.thresholds:
            self.threshold_to_metrics[threshold] = self.manager.dict({
                'conf_tp': 0,
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
        feature_path = f'{self.features_dir}{root}.hdf5'
        labels_path = f'{self.labels_dir}{root}.hdf5'
        try:
            with h5py.File(feature_path, 'r') as feat_f, h5py.File(labels_path, 'r') as labels_f:
                labels = labels_f['labels'][:]
                confidences = labels_f['confidences'][:]

                rank = feat_f['rank'][:]

                edges = feat_f['edges'][:]

                box_cutoff_nodes = rank < self.box_cutoff

                if len(labels) > self.fov:
                    indices = np.where(rank < self.fov)[0]
                    labels = labels[indices]
                    confidences = confidences[indices]
                    edges = prune_edges(edges, indices)
                    box_cutoff_nodes = box_cutoff_nodes[indices]

                assert(len(labels) == len(output))

                # Might need to change this, not sure
                threshold_to_output = {}
                for threshold in self.thresholds:
                    threshold_to_output[threshold] = np.where(output < threshold, False, True)

                g = nx.Graph()
                g.add_edges_from(edges)

                label_ccs = get_label_components(g, labels, box_cutoff_nodes)
                
                new_label_ccs, confidences = remove_isolated_errors(g, label_ccs, confidences)
                label_ccs = new_label_ccs

                for threshold in self.thresholds:
                    output_ccs = get_output_components(g, threshold_to_output[threshold], confidences)
                    output_ccs, conf_fp = remove_big_output_ccs(output_ccs, labels, self.obj_det_error_cloud_ratio, box_cutoff_nodes)
                    conf_tp, conf_fn = count_shared_and_unshared(label_ccs, output_ccs)
                    # if threshold == 0.5:
                    #     if conf_fn > 0:
                    #         print("root with missed error", root)
                    self.threshold_to_metrics[threshold]['conf_tp'] += conf_tp
                    self.threshold_to_metrics[threshold]['conf_fn'] += conf_fn
                    self.threshold_to_metrics[threshold]['conf_fp'] += conf_fp

        except Exception as e:
            print("root: ", root, "error: ", e)
            return None
            
        return 1 

# If any error locations are isolated with no confident non errors nearby 
# then remove from label cc list and mark them as unconfident so they get ignored in output prediction
def remove_isolated_errors(graph, label_ccs, confidences):
    new_label_ccs = []
    for cc in label_ccs:
        is_isolated = True
        for node in cc:
            for neighbor in graph.adj[node]:
                if neighbor not in cc and confidences[neighbor] == True:
                    is_isolated = False
                    break
            if not is_isolated:
                break
        if is_isolated:
            for node in cc:
                confidences[node] = True
        else:
            new_label_ccs.append(cc)
    return new_label_ccs, confidences

def get_label_components(graph, labels, box_cutoff_nodes):
    # data=False since we don't need other attributes of the nodes after we do this
    subgraph_nodes = [i for i in range(len(labels)) if (labels[i] == True and box_cutoff_nodes[i] == True)]
    subgraph = graph.subgraph(subgraph_nodes)
    ccs = list(nx.connected_components(subgraph))
    return ccs

def get_output_components(graph, threshold_output, confidences):
    subgraph_nodes = [i for i in range(len(confidences)) if (threshold_output[i] == True and confidences[i] == True)]
    subgraph = graph.subgraph(subgraph_nodes)
    ccs = list(nx.connected_components(subgraph))
    return ccs

# Removes output ccs that have too high of an output ratio and removes nodes that aren't in box cutoff after deciding if the cc should be removed
def remove_big_output_ccs(output_ccs, labels, obj_det_error_cloud_ratio, box_cutoff_nodes):
    fp = 0
    new_output_ccs = []
    for cc in output_ccs:
        total = len(cc)
        count = 0
        new_cc = set()
        for node in cc:
            if labels[node] == True: # TODO: Should I do this or no
                count += 1
            if box_cutoff_nodes[node]:
                new_cc.add(node)
        ratio = count / total
        if ratio >= obj_det_error_cloud_ratio:
            if len(new_cc) > 0:
                new_output_ccs.append(new_cc)
        if ratio == 0:
            fp += 1
    return new_output_ccs, fp

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

def obj_det_dataloader(config, dataset):
    num_workers = config['loader']['num_workers']
    batch_size = config['loader']['batch_size']
    shuffle = False # Don't need to shuffle since its the same set
    pin_memory = False # cpu only so don't need it
    persistent_workers= False
    prefetch_factor = None

    return DataLoader(
            dataset=dataset,
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor)

def obj_det_plot(metrics_dict, thresholds, epoch, save_dir, obj_det_error_cloud_ratio, box_cutoff):
    recalls = []
    precisions = []
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        # Using label tp instead of output tp for now
        precision, recall = get_precision_and_recall(metrics_dict[threshold]['conf_tp'], metrics_dict[threshold]['conf_fn'], metrics_dict[threshold]['conf_fp'])
        precisions.append(precision)
        recalls.append(recall)

    plt.figure(figsize=(8, 8))
    plt.plot(recalls, precisions, marker='.', markersize=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Object Precision-Recall Curve with Cloud Ratio: {obj_det_error_cloud_ratio} in Box: {box_cutoff} at Epoch: {epoch}')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    for i in range(len(thresholds)):
        plt.scatter(recalls[i], precisions[i], c='red', s=30, label=f'Recall: {recalls[i]:.2f}, Precision: {precisions[i]:.2f}, Threshold: {thresholds[i]}')  # Mark with a red dot
    plt.legend()
    plt.grid(True)
    save_path = f'{save_dir}obj_precision_recall_{epoch}.png'
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
    ckpt_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/ckpt/"   
    run_id = 'AUT-255'
    run_dir = f'{ckpt_dir}{run_id}/'
    epoch = 30
    ckpt_path = f'{run_dir}model_{epoch}.pt'
    # with open(f'{run_dir}config.json', 'r') as f:
    #     config = json.load(f)
    config = data_utils.get_config('base')
    
    model = create_model(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # roots = [864691135463333789]
    # roots = [864691136379030485]
    # roots = [864691136443843459]
    # roots = [864691136443843459, 864691135463333789]
    # roots = ['864691136041340246_000']
    # roots = ['864691136521643153_000']
    # roots = ['864691135463333789_000']
    # roots = ['864691135439772402_000']

    mode = 'val'
    data = AutoProofDataset(config, mode)
    config['loader']['batch_size'] = 32
    config['loader']['num_workers'] = 32
    config['data']['obj_det_val_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_379668/val_conf_no_error_in_box_roots.txt"
    config['trainer']['obj_det_error_cloud_ratio'] = 0.2
    config['data']['box_cutoff'] = 100
    data_loader = build_dataloader(config, data, mode)

    model.eval()
    with torch.no_grad():
        root_to_output = {}

        # for root in roots:
        #     print("Getting root output")
        #     # Remember this doesn't pre mask so only use things above fov for testing right now
        #     vertices, edges, labels, confidences, output, _, _, _, _ = get_root_output(model, device, data, root)
        #     output = output.detach().cpu().numpy().squeeze(-1)
        #     root_to_output[root] = output

        obj_det_val_roots = set(data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_379668/val_conf_no_error_in_box_roots.txt"))

        with tqdm(total=len(data) / config['loader']['batch_size'], desc=mode) as pbar:
            for i, data in enumerate(data_loader):
                # root (b, 1) input (b, fov, d), labels (b, fov, 1), conf (b, fov, 1), adj (b, fov, fov)
                roots, input, labels, confidences, dist_to_error, rank, adj = data
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
                # roots = roots.detach().cpu().numpy().astype(int)
                for i in range(len(roots)):
                    if roots[i] in obj_det_val_roots:
                        root_to_output[roots[i]] = output[i][mask[i]]
                pbar.update()
        
        print("creating obj dataset")
        obj_det_data = ObjectDetectionDataset(config, root_to_output)
        obj_det_loader = obj_det_dataloader(config, obj_det_data)
        start_time = time.time()
        with tqdm(total=(len(obj_det_data) / config['loader']['batch_size']), desc="obj det all") as pbar:
            for i, data in enumerate(obj_det_loader):
                pbar.update()
        end_time = time.time()
        print("time", end_time - start_time)
        
        metrics_dict = obj_det_data.__getmetrics__()
        print("metrics at threshold 0.01", obj_det_data.__getmetrics__()[0.01])
        print("metrics at threshold 0.4", obj_det_data.__getmetrics__()[0.4])
        print("metrics at threshold 0.5", obj_det_data.__getmetrics__()[0.5])
        print("metrics at threshold 0.6", obj_det_data.__getmetrics__()[0.6])
        print("metrics at threshold 0.99", obj_det_data.__getmetrics__()[0.99])

        # save_dir = run_dir
        save_dir = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/obj_test/'

        save_path = obj_det_plot(metrics_dict, config['trainer']['thresholds'], epoch, save_dir, config['trainer']['obj_det_error_cloud_ratio'], config['data']['box_cutoff'])
    print("done")    

