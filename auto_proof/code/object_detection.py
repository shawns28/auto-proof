from auto_proof.code.dataset import AutoProofDataset, adjency_to_edge_list_torch_skip_diag, prune_edges, build_dataloader
from auto_proof.code.pre import data_utils
from auto_proof.code.model import create_model
from auto_proof.code.visualize import get_root_output

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
        self.branch_degrees = config['trainer']['branch_degrees']
        self.thresholds = config['trainer']['thresholds']
        self.obj_det_error_cloud_ratios = config['trainer']['obj_det_error_cloud_ratios']

        self.roots = list(root_to_output.keys())
        self.manager = Manager()
        self.root_to_output = self.manager.dict(root_to_output)
        self.threshold_to_metrics = self.manager.dict()
        for threshold in self.thresholds:
            for cloud_ratio in self.obj_det_error_cloud_ratios:
                self.threshold_to_metrics[f'{threshold}_{cloud_ratio}'] = self.manager.dict({
                    'conf_tp': 0,
                    'conf_fn': 0, 
                    'conf_fp': 0, 
                })
        for degree in self.branch_degrees:
            self.threshold_to_metrics[degree] = self.manager.dict({
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
                if len(labels) < 20:
                    return '', False, False
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
                labels = labels.astype(bool)
                confidences = confidences.astype(bool)
                assert(len(labels) == len(output))

                # Might need to change this, not sure
                threshold_to_output = {}
                for threshold in self.thresholds:
                    threshold_to_output[threshold] = np.where(output > threshold, True, False)

                g = nx.Graph()
                g.add_nodes_from(range(len(labels)))
                g.add_edges_from(edges)

                degree_to_output = {}
                node_degrees = g.degree()
                node_degrees = np.array([degree for _, degree in node_degrees])
                # print("node degrees", node_degrees)
                for degree in self.branch_degrees:
                    degree_to_output[degree] = np.where(node_degrees >= degree, True, False)
                # print("degree_to_output", degree_to_output)

                label_ccs = get_label_components(g, labels, box_cutoff_nodes)
                
                label_ccs, confidences = remove_isolated_errors(g, label_ccs, confidences)
                missed_error = False
                found_error = False
                for threshold in self.thresholds:
                    output_ccs = get_output_components(g, threshold_to_output[threshold], confidences, box_cutoff_nodes)
                    for cloud_ratio in self.obj_det_error_cloud_ratios:
                        curr_output_ccs, conf_fp, conf_fn = remove_big_output_ccs(output_ccs, labels, cloud_ratio)
                        conf_tp, missed_fn = count_shared_and_unshared(label_ccs, curr_output_ccs)
                        conf_fn += missed_fn

                        self.threshold_to_metrics[f'{threshold}_{cloud_ratio}']['conf_tp'] += conf_tp
                        self.threshold_to_metrics[f'{threshold}_{cloud_ratio}']['conf_fn'] += conf_fn
                        self.threshold_to_metrics[f'{threshold}_{cloud_ratio}']['conf_fp'] += conf_fp

                        # For baseline
                        if (threshold == 0.05 and cloud_ratio == 0.1) and conf_fn > 0:
                            missed_error = True
                        elif (threshold == 0.05 and cloud_ratio == 0.1) and conf_tp > 0:
                            found_error = True

                        # For no segclr
                        # if (threshold == 0.1 and cloud_ratio == 0.1) and conf_fn > 0:
                        #     missed_error = True
                        # elif (threshold == 0.1 and cloud_ratio == 0.1) and conf_tp > 0:
                        #     found_error = True

                        # For including everything in core
                        # if (threshold == 0.05 and cloud_ratio == 0.1) and conf_fn > 0:
                        #     missed_error = True
                        # elif (threshold == 0.05 and cloud_ratio == 0.1) and conf_tp > 0:
                        #     found_error = True
                for degree in self.branch_degrees:
                    degree_ccs = get_output_components(g, degree_to_output[degree], confidences, box_cutoff_nodes)
                    conf_tp, conf_fn = count_shared_and_unshared(label_ccs, degree_ccs)
                    _, conf_fp = count_shared_and_unshared(degree_ccs, label_ccs)
                    self.threshold_to_metrics[degree]['conf_tp'] += conf_tp
                    self.threshold_to_metrics[degree]['conf_fn'] += conf_fn
                    self.threshold_to_metrics[degree]['conf_fp'] += conf_fp                  
                
                return root, missed_error, found_error
        except Exception as e:
            print("root: ", root, "error: ", e)
            return None, False, False
            
        return '', False, False 

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
                confidences[node] = False
        else:
            new_label_ccs.append(cc)
    return new_label_ccs, confidences

def get_label_components(graph, labels, box_cutoff_nodes):
    # data=False since we don't need other attributes of the nodes after we do this
    subgraph_nodes = [i for i in range(len(labels)) if (labels[i] == True and box_cutoff_nodes[i] == True)]
    subgraph = graph.subgraph(subgraph_nodes)
    ccs = list(nx.connected_components(subgraph))
    return ccs

def get_output_components(graph, output, confidences, box_cutoff_nodes):
    # NOTE: Trying it withoout the confidences only check
    subgraph_nodes = [i for i in range(len(confidences)) if ((output[i] == True and confidences[i] == True) and box_cutoff_nodes[i] == True)]
    # subgraph_nodes = [i for i in range(len(confidences)) if (output[i] == True and box_cutoff_nodes[i] == True)]
    subgraph = graph.subgraph(subgraph_nodes)
    ccs = list(nx.connected_components(subgraph))
    return ccs

# Removes output ccs that have too high of an output ratio and removes nodes that aren't in box cutoff after deciding if the cc should be removed
def remove_big_output_ccs(output_ccs, labels, obj_det_error_cloud_ratio):
    fp = 0
    fn = 0
    new_output_ccs = []
    for cc in output_ccs:
        total = len(cc)
        count = 0
        for node in cc:
            if labels[node] == True: # TODO: Should I do this or no
                count += 1
        ratio = count / total
        if ratio >= obj_det_error_cloud_ratio:
            new_output_ccs.append(cc) 
        elif ratio == 0:    
            fp += 1
        else: # ratio between 0 and cloud_ratio, not specific enough cloud
            fn += 1
    return new_output_ccs, fp, fn

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

def obj_det_plot(metrics_dict, thresholds, epoch, save_dir, cloud_ratio, box_cutoff, branch_degrees):
    recalls = []
    precisions = []
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        # Using label tp instead of output tp for now
        precision, recall = get_precision_and_recall(metrics_dict[f'{threshold}_{cloud_ratio}']['conf_tp'], metrics_dict[f'{threshold}_{cloud_ratio}']['conf_fn'], metrics_dict[f'{threshold}_{cloud_ratio}']['conf_fp'])
        precisions.append(precision)
        recalls.append(recall)

    plt.figure(figsize=(8, 8))
    plt.plot(recalls, precisions, marker='.', markersize=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    # plt.title(f'Object Precision-Recall Curve with Cloud Ratio: {cloud_ratio} in Box: {box_cutoff} at Epoch: {epoch}')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    for i in range(len(thresholds)):
        plt.scatter(recalls[i], precisions[i], c='red', s=30, label=f'Recall: {recalls[i]:.2f}, Precision: {precisions[i]:.2f}, Threshold: {thresholds[i]}')  # Mark with a red dot
    
    # recalls = []
    # precisions = []
    # for branch_degree in branch_degrees:
    #     # print("branch degree", branch_degree)
    #     # Using label tp instead of output tp for now
    #     precision, recall = get_precision_and_recall(metrics_dict[branch_degree]['conf_tp'], metrics_dict[branch_degree]['conf_fn'], metrics_dict[branch_degree]['conf_fp'])
    #     precisions.append(precision)
    #     recalls.append(recall)
    # # print(precisions)
    # # print(recalls)
    # for i in range(len(branch_degrees)):
    #     plt.scatter(recalls[i], precisions[i], c='blue', s=30, label=f'Recall: {recalls[i]:.2f}, Precision: {precisions[i]:.2f}, Degree: {branch_degrees[i]}')  # Mark with a red dot

    # plt.legend()
    plt.grid(True)
    save_path = f'{save_dir}obj_precision_recall_{epoch}_{cloud_ratio}.png'
    plt.savefig(save_path)
    return save_path

def obj_det_plot_legend(metrics_dict, thresholds, epoch, save_dir, cloud_ratio, box_cutoff, branch_degrees):
    recalls = []
    precisions = []
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        # Using label tp instead of output tp for now
        precision, recall = get_precision_and_recall(metrics_dict[f'{threshold}_{cloud_ratio}']['conf_tp'], metrics_dict[f'{threshold}_{cloud_ratio}']['conf_fn'], metrics_dict[f'{threshold}_{cloud_ratio}']['conf_fp'])
        precisions.append(precision)
        recalls.append(recall)

    plt.figure(figsize=(8, 8))
    plt.plot(recalls, precisions, marker='.', markersize=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    # plt.title(f'Object Precision-Recall Curve with Cloud Ratio: {cloud_ratio} in Box: {box_cutoff} at Epoch: {epoch}')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    for i in range(len(thresholds)):
        plt.scatter(recalls[i], precisions[i], c='red', s=30, label=f'Recall: {recalls[i]:.2f}, Precision: {precisions[i]:.2f}, Threshold: {thresholds[i]}')  # Mark with a red dot
    
    # recalls = []
    # precisions = []
    # for branch_degree in branch_degrees:
    #     # print("branch degree", branch_degree)
    #     # Using label tp instead of output tp for now
    #     precision, recall = get_precision_and_recall(metrics_dict[branch_degree]['conf_tp'], metrics_dict[branch_degree]['conf_fn'], metrics_dict[branch_degree]['conf_fp'])
    #     precisions.append(precision)
    #     recalls.append(recall)
    # # print(precisions)
    # # print(recalls)
    # for i in range(len(branch_degrees)):
    #     plt.scatter(recalls[i], precisions[i], c='blue', s=30, label=f'Recall: {recalls[i]:.2f}, Precision: {precisions[i]:.2f}, Degree: {branch_degrees[i]}')  # Mark with a red dot

    plt.legend()
    plt.grid(True)
    save_path = f'{save_dir}obj_precision_recall_{epoch}_{cloud_ratio}_ledeng.png'
    plt.savefig(save_path)
    return save_path

def plot_branch_statistics(metrics_dict, branch_degrees, epoch, save_dir, cloud_ratio, box_cutoff):
    """
    Generates and saves a precision-recall plot specifically for branch statistics.

    Args:
        metrics_dict (dict): A dictionary containing 'conf_tp', 'conf_fn', and 'conf_fp'
                             for different branch degrees.
        branch_degrees (list): A list of branch degree values to plot.
        epoch (int): The current epoch number, used in the plot title and filename.
        save_dir (str): The directory where the plot will be saved.
        cloud_ratio (float): The cloud ratio, used in the plot title and filename.
        box_cutoff (float): The box cutoff value, used in the plot title.
    """
    recalls = []
    precisions = []

    # Assuming get_precision_and_recall is defined elsewhere and works correctly
    # def get_precision_and_recall(tp, fn, fp):
    #     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    #     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    #     return precision, recall

    for branch_degree in branch_degrees:
        # It's good practice to handle cases where a branch_degree might be missing
        if branch_degree in metrics_dict:
            precision, recall = get_precision_and_recall(
                metrics_dict[branch_degree]['conf_tp'],
                metrics_dict[branch_degree]['conf_fn'],
                metrics_dict[branch_degree]['conf_fp']
            )
            precisions.append(precision)
            recalls.append(recall)
        else:
            print(f"Warning: No metrics found for branch degree: {branch_degree}")
            recalls.append(0) # or some other default
            precisions.append(0) # or some other default

    plt.figure(figsize=(8, 8))
    
    # Plotting the line connecting the points
    plt.plot(recalls, precisions, linestyle='-', marker='o', color='blue', label='Branch Precision-Recall Curve')

    # Plotting individual points with labels
    for i in range(len(branch_degrees)):
        plt.scatter(recalls[i], precisions[i], c='red', s=50, zorder=5, # zorder to ensure dots are on top of the line
                    label=f'Degree: {branch_degrees[i]}, P: {precisions[i]:.2f}, R: {recalls[i]:.2f}')

    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    # plt.title(f'Branch Precision-Recall Curve with Cloud Ratio: {cloud_ratio} in Box: {box_cutoff} at Epoch: {epoch}', fontsize=16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.legend(fontsize=10)
    plt.grid(True)
    
    save_path = f'{save_dir}branch_precision_recall_{epoch}_{cloud_ratio}.png'
    plt.savefig(save_path)
    plt.close() # Close the plot to free up memory
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
    run_id = 'AUT-275' # first segclr
    run_id = 'AUT-301'
    run_id = 'AUT-322'
    run_id = 'AUT-330' # baseline
    # run_id = 'AUT-331' # no segclr
    run_dir = f'{ckpt_dir}{run_id}/'
    # epoch = 40
    epoch = 60
    ckpt_path = f'{run_dir}model_{epoch}.pt'
    with open(f'{run_dir}config.json', 'r') as f:
        config = json.load(f)
    # config = data_utils.get_config('base')
    if run_id == 'AUT-330':
        config['data']['labels_dir'] = "labels_at_1300_ignore_inbetween/"
    
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
    # config['loader']['normalize'] = False
    config['loader']['num_workers'] = 32
    config['trainer']['obj_det_error_cloud_ratios'] = [0.1, 0.2, 0.3]
    config['data']['all_roots'] = "all_roots_og.txt"
    config['data']['train_roots'] = "train_roots_og.txt"
    config['data']['val_roots'] = "val_roots_og.txt"
    config['data']['test_roots'] = "test_roots_og.txt"

    mode = 'train'
    data = AutoProofDataset(config, mode)
    # config['loader']['batch_size'] = 32
    # config['data']['obj_det_val_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_379668/val_conf_no_error_in_box_roots.txt"
    # config['data']['obj_det_val_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/shared_conf_no_error_in_box_roots.txt"
    # config['data']['obj_det_val_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/sharedval_conf_no_error_in_box_roots.txt"
    # config['data']['obj_det_val_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_503534/val_conf_no_error_in_box_roots.txt"
    # config['data']['obj_det_val_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_503534/train_conf_no_error_in_box_roots.txt"
    # config['trainer']['obj_det_error_cloud_ratio'] = 0.2
    # config['data']['box_cutoff'] = 100
    data_loader = build_dataloader(config, data, mode)
    # config['trainer']['branch_degrees'] = [3, 4, 5]

    model.eval()
    with torch.no_grad():
        root_to_output = {}

        # config['loader']['num_workers'] = 1
        # roots = ['864691135657412579_000']
        # for root in roots:
        #     print("Getting root output")
        #     # Remember this doesn't pre mask so only use things above fov for testing right now
        #     vertices, edges, labels, confidences, output, _, _, _, _, _ , _, _ = get_root_output(model, device, data, root)
        #     output = output.detach().cpu().numpy().squeeze(-1)
        #     root_to_output[root] = output

        # obj_det_roots = set(data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_503534/val_conf_no_error_in_box_roots.txt"))
        # obj_det_roots = set(data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_503534/train_conf_no_error_in_box_roots.txt"))
        obj_det_roots = set(data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/roots_343_1300/split_598963/train_conf_no_error_in_box_roots_og.txt"))

        with tqdm(total=len(data) / config['loader']['batch_size'], desc=mode) as pbar:
            for i, data in enumerate(data_loader):
                # root (b, 1) input (b, fov, d), labels (b, fov, 1), conf (b, fov, 1), adj (b, fov, fov)
                roots, input, labels, confidences, dist_to_error, rank, adj, _ = data
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
                    if roots[i] in obj_det_roots:
                        root_to_output[roots[i]] = output[i][mask[i]]
                pbar.update()
        
        print("creating obj dataset")
        obj_det_data = ObjectDetectionDataset(config, root_to_output)
        obj_det_loader = obj_det_dataloader(config, obj_det_data)
        start_time = time.time()
        missed_roots = []
        found_roots = []
        other_roots = []
        with tqdm(total=(len(obj_det_data) / config['loader']['batch_size']), desc="obj det all") as pbar:
            for i, data in enumerate(obj_det_loader):
                roots, missed, found = data
                for i, root in enumerate(roots):
                    if missed[i]:
                        missed_roots.append(root)
                    elif found[i]:
                        found_roots.append(root)
                    else:
                        other_roots.append(root)
                
                pbar.update()
        end_time = time.time()
        # print("time", end_time - start_time)
        # print("missed_roots", missed_roots)
        save_dir = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/plots_before_pres/'
        data_utils.save_txt(f"{save_dir}missed.txt", missed_roots)
        data_utils.save_txt(f"{save_dir}found.txt", found_roots)
        data_utils.save_txt(f"{save_dir}other.txt", other_roots)
        metrics_dict = obj_det_data.__getmetrics__()
        
        # branch model
        cloud_ratio = 0.1
        save_path = plot_branch_statistics(metrics_dict, config['trainer']['branch_degrees'], epoch, save_dir, cloud_ratio, config['data']['box_cutoff'])

        # for cloud_ratio in config['trainer']['obj_det_error_cloud_ratios']:
        save_path = obj_det_plot(metrics_dict, config['trainer']['thresholds'], epoch, save_dir, cloud_ratio, config['data']['box_cutoff'], config['trainer']['branch_degrees'])
        save_path = obj_det_plot_legend(metrics_dict, config['trainer']['thresholds'], epoch, save_dir, cloud_ratio, config['data']['box_cutoff'], config['trainer']['branch_degrees'])

        print("metrics at threshold 0.01", obj_det_data.__getmetrics__()["0.01_0.2"])
        print("metrics at threshold 0.4", obj_det_data.__getmetrics__()["0.4_0.2"])
        print("metrics at threshold 0.5", obj_det_data.__getmetrics__()["0.5_0.2"])
        print("metrics at threshold 0.99", obj_det_data.__getmetrics__()["0.99_0.2"])

        print("metrics at degree 3", obj_det_data.__getmetrics__()[3])

    print("done")