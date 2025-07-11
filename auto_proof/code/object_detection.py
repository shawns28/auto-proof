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
    """
    A PyTorch Dataset for evaluating object detection metrics predictions.

    This dataset takes model outputs and ground truth labels, then calculates
    precision-recall metrics for object-level error detection, considering
    connected components (error clouds) and various thresholds.
    """

    def __init__(self, config, root_to_output):
        """
        Initializes the ObjectDetectionDataset.

        Args:
            config (dict): A dictionary containing configuration parameters for
                           data paths, loader settings, and trainer parameters.
            root_to_output (dict): A dictionary mapping root IDs (str) to
                                   their corresponding model output predictions (np.ndarray).
        """
        self.config = config

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
        """
        Returns the total number of roots in the dataset.

        Returns:
            int: The number of roots.
        """
        return len(self.roots)

    def __getmetrics__(self):
        """
        Returns a dictionary of accumulated object detection metrics.

        Returns:
            dict: A copy of the manager dictionary containing TP, FN, FP counts
                  for various thresholds and branch degrees.
        """
        return dict(self.threshold_to_metrics)

    def __getitem__(self, index):
        """
        Processes a single root's data to compute object detection metrics.

        Loads features and labels for a root, prunes data to FOV, calculates
        connected components for ground truth errors and model predictions,
        and updates shared metric counters.

        Note: Only counts roots that have 10 times confident nodes as error nodes
              and total node count >= 20.

        Args:
            index (int): The index of the root to process.

        Returns:
            tuple: A tuple containing:
                - root (str): The ID of the processed root.
                - missed_error (bool): True if a ground truth error was missed by the baseline criteria.
                - found_error (bool): True if a ground truth error was found by the baseline criteria.
                - ignored (bool): True if the root was ignored due to data conditions (e.g., too few nodes).
            Returns empty string and False, False, False if an unexpected error occurs.
        """
        root = self.roots[index]
        output = self.root_to_output[root]
        feature_path = f'{self.features_dir}{root}.hdf5'
        labels_path = f'{self.labels_dir}{root}.hdf5'
        try:
            with h5py.File(feature_path, 'r') as feat_f, h5py.File(labels_path, 'r') as labels_f:
                labels = labels_f['labels'][:]
                confidences = labels_f['confidences'][:]

                if len(labels) < 20:
                    return root, False, False, True

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

                # Calculate number of confident nodes within box cutoff
                num_confident_in_box_cutoff = np.sum(confidences[box_cutoff_nodes])
                num_errors = np.sum(labels)

                # Only counts roots that have 10 times confident nodes as error nodes
                if num_confident_in_box_cutoff < 10 * num_errors:
                    return root, False, False, True

                # Might need to change this, not sure
                threshold_to_output = {}
                for threshold in self.thresholds:
                    threshold_to_output[threshold] = np.where(output > threshold, True, False)

                g = nx.Graph()
                g.add_nodes_from(range(len(labels)))
                if len(edges) > 0: # Ensure edges exist before adding
                    g.add_edges_from(edges)

                degree_to_output = {}
                node_degrees = g.degree()
                node_degrees = np.array([degree for _, degree in node_degrees])
                for degree in self.branch_degrees:
                    degree_to_output[degree] = np.where(node_degrees >= degree, True, False)

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

                for degree in self.branch_degrees:
                    degree_ccs = get_output_components(g, degree_to_output[degree], confidences, box_cutoff_nodes)
                    conf_tp, conf_fn = count_shared_and_unshared(label_ccs, degree_ccs)
                    _, conf_fp = count_shared_and_unshared(degree_ccs, label_ccs)
                    self.threshold_to_metrics[degree]['conf_tp'] += conf_tp
                    self.threshold_to_metrics[degree]['conf_fn'] += conf_fn
                    self.threshold_to_metrics[degree]['conf_fp'] += conf_fp

                return root, missed_error, found_error, False
        except Exception as e:
            print("root: ", root, "error: ", e)
            # If an error occurs, consider it "ignored" for the purpose of the metrics
            return root, False, False, False

        return '', False, False, False

def remove_isolated_errors(graph, label_ccs, confidences):
    """
    Identifies and removes isolated error components from the label CCs.

    An error component is considered "isolated" if none of its nodes are
    adjacent to a 'confident' non-error node. Isolated errors are then
    marked as unconfident in the `confidences` array and removed from
    the `label_ccs` list, so they are not considered for TP/FN calculations.

    Args:
        graph (nx.Graph): The NetworkX graph representing the neuron skeleton.
        label_ccs (list of sets): A list where each set represents the nodes
                                  of a connected component of ground truth errors.
        confidences (np.ndarray): A boolean numpy array indicating which nodes are
                                 'confident' (True) or 'unconfident' (False).

    Returns:
        tuple: A tuple containing:
            - new_label_ccs (list of sets): The updated list of label connected components,
                                            with isolated ones removed.
            - confidences (np.ndarray): The updated confidences array, with isolated
                                        error nodes marked as False.
    """
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
    """
    Identifies connected components (CCs) of ground truth error nodes within the box cutoff.

    Args:
        graph (nx.Graph): The NetworkX graph representing the neuron skeleton.
        labels (np.ndarray): A boolean numpy array indicating ground truth error nodes (True).
        box_cutoff_nodes (np.ndarray): A boolean numpy array indicating nodes within the box cutoff.

    Returns:
        list of sets: A list where each set contains the nodes forming a
                      connected component of error nodes.
    """
    subgraph_nodes = [i for i in range(len(labels)) if (labels[i] == True and box_cutoff_nodes[i] == True)]
    subgraph = graph.subgraph(subgraph_nodes)
    ccs = list(nx.connected_components(subgraph))
    return ccs

def get_output_components(graph, output, confidences, box_cutoff_nodes):
    """
    Identifies connected components (CCs) of predicted error nodes.

    These are nodes that are predicted as True, are confident, and are within the box cutoff.

    Args:
        graph (nx.Graph): The NetworkX graph representing the neuron skeleton.
        output (np.ndarray): A boolean numpy array representing model predictions (True for error).
        confidences (np.ndarray): A boolean numpy array indicating confident nodes.
        box_cutoff_nodes (np.ndarray): A boolean numpy array indicating nodes within the box cutoff.

    Returns:
        list of sets: A list where each set contains the nodes forming a
                      connected component of predicted error nodes.
    """
    subgraph_nodes = [i for i in range(len(confidences)) if ((output[i] == True and confidences[i] == True) and box_cutoff_nodes[i] == True)]
    subgraph = graph.subgraph(subgraph_nodes)
    ccs = list(nx.connected_components(subgraph))
    return ccs

# Removes output ccs that have too high of an output ratio and removes nodes that aren't in box cutoff after deciding if the cc should be removed
def remove_big_output_ccs(output_ccs, labels, obj_det_error_cloud_ratio):
    """
    Filters predicted error connected components based on their "error ratio".

    An output CC is kept if the ratio of its nodes that are true errors (according to `labels`)
    is greater than or equal to `obj_det_error_cloud_ratio`. CCs that fail this criterion
    are counted as False Positives (FP) if their ratio is 0, or False Negatives (FN)
    if their ratio is between 0 and `obj_det_error_cloud_ratio`.

    Args:
        output_ccs (list of sets): A list of connected components of predicted error nodes.
        labels (np.ndarray): A boolean numpy array indicating ground truth error nodes.
        obj_det_error_cloud_ratio (float): The minimum ratio of true error nodes within
                                            an output CC for it to be considered a valid detection.

    Returns:
        tuple: A tuple containing:
            - new_output_ccs (list of sets): The filtered list of valid output connected components.
            - fp (int): Count of False Positives (output CCs with 0 true error nodes).
            - fn (int): Count of False Negatives (output CCs with some true error nodes, but ratio too low).
    """
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
    """
    Compares two lists of sets (connected components) to count shared and unshared components.

    A component from `list_of_sets1` is considered "shared" if it has at least one
    common element with any component in `list_of_sets2`. Otherwise, it's "unshared".

    This function is typically used to calculate True Positives (shared) and False Negatives (unshared)
    when `list_of_sets1` represents ground truth components and `list_of_sets2` represents
    predicted components.

    Args:
        list_of_sets1 (list of sets): The first list of connected components.
        list_of_sets2 (list of sets): The second list of connected components to compare against.

    Returns:
        tuple: A tuple containing:
            - shared (int): The count of components from `list_of_sets1` that overlap with `list_of_sets2`.
            - unshared (int): The count of components from `list_of_sets1` that do not overlap with `list_of_sets2`.
    """
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
    """
    Builds a PyTorch DataLoader specifically for object detection evaluation.

    This DataLoader is configured for evaluation, meaning shuffling is disabled
    and pin_memory/persistent_workers are typically set to False for CPU-only processing.

    Args:
        config (dict): A dictionary containing configuration parameters,
                       specifically 'loader' settings for num_workers and batch_size.
        dataset (torch.utils.data.Dataset): The dataset to load (e.g., ObjectDetectionDataset).

    Returns:
        torch.utils.data.DataLoader: A configured PyTorch DataLoader.
    """
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
    """
    Generates and saves an object detection Precision-Recall (P-R) curve.

    The plot shows Precision vs. Recall for various prediction thresholds.

    Args:
        metrics_dict (dict): A dictionary containing 'conf_tp', 'conf_fn', and 'conf_fp'
                             for different threshold and cloud ratio combinations.
        thresholds (list of float): A list of prediction probability thresholds.
        epoch (int): The current epoch number, used in the plot title and filename.
        save_dir (str): The directory where the plot will be saved.
        cloud_ratio (float): The error cloud ratio used, included in the plot title and filename.
        box_cutoff (int): The box cutoff value, included in the plot title.
        branch_degrees (list of int): Not directly used in this plot, but kept for signature consistency
                                      if needed in future variations.

    Returns:
        str: The file path where the plot was saved.
    """
    recalls = []
    precisions = []
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        precision, recall = get_precision_and_recall(metrics_dict[f'{threshold}_{cloud_ratio}']['conf_tp'], metrics_dict[f'{threshold}_{cloud_ratio}']['conf_fn'], metrics_dict[f'{threshold}_{cloud_ratio}']['conf_fp'])
        precisions.append(precision)
        recalls.append(recall)

    plt.figure(figsize=(8, 8))
    plt.plot(recalls, precisions, marker='.', markersize=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title(f'Object Precision-Recall Curve with Cloud Ratio: {cloud_ratio} in Box: {box_cutoff} at Epoch: {epoch}')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    for i in range(len(thresholds)):
        plt.scatter(recalls[i], precisions[i], c='red', s=30, label=f'Recall: {recalls[i]:.2f}, Precision: {precisions[i]:.2f}, Threshold: {thresholds[i]}')  # Mark with a red dot
    
    plt.grid(True)
    save_path = f'{save_dir}obj_precision_recall_{epoch}_{cloud_ratio}.png'
    plt.savefig(save_path)
    return save_path

def obj_det_plot_legend(metrics_dict, thresholds, epoch, save_dir, cloud_ratio, box_cutoff, branch_degrees):
    """
    Generates and saves an object detection Precision-Recall (P-R) curve with a full legend.

    This function is similar to `obj_det_plot` but specifically includes detailed labels
    for each point in the legend.

    Args:
        metrics_dict (dict): A dictionary containing 'conf_tp', 'conf_fn', and 'conf_fp'
                             for different threshold and cloud ratio combinations.
        thresholds (list of float): A list of prediction probability thresholds.
        epoch (int): The current epoch number, used in the plot title and filename.
        save_dir (str): The directory where the plot will be saved.
        cloud_ratio (float): The error cloud ratio used, included in the plot title and filename.
        box_cutoff (int): The box cutoff value, included in the plot title.
        branch_degrees (list of int): The branch degrees to use for metrcis.

    Returns:
        str: The file path where the plot was saved.
    """
    recalls = []
    precisions = []
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        precision, recall = get_precision_and_recall(metrics_dict[f'{threshold}_{cloud_ratio}']['conf_tp'], metrics_dict[f'{threshold}_{cloud_ratio}']['conf_fn'], metrics_dict[f'{threshold}_{cloud_ratio}']['conf_fp'])
        precisions.append(precision)
        recalls.append(recall)

    plt.figure(figsize=(8, 8))
    plt.plot(recalls, precisions, marker='.', markersize=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title(f'Object Precision-Recall Curve with Cloud Ratio: {cloud_ratio} in Box: {box_cutoff} at Epoch: {epoch}')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    for i in range(len(thresholds)):
        plt.scatter(recalls[i], precisions[i], c='red', s=30, label=f'Recall: {recalls[i]:.2f}, Precision: {precisions[i]:.2f}, Threshold: {thresholds[i]}')  # Mark with a red dot
    
    recalls = []
    precisions = []
    for branch_degree in branch_degrees:
        precision, recall = get_precision_and_recall(metrics_dict[branch_degree]['conf_tp'], metrics_dict[branch_degree]['conf_fn'], metrics_dict[branch_degree]['conf_fp'])
        precisions.append(precision)
        recalls.append(recall)

    for i in range(len(branch_degrees)):
        plt.scatter(recalls[i], precisions[i], c='blue', s=30, label=f'Recall: {recalls[i]:.2f}, Precision: {precisions[i]:.2f}, Degree: {branch_degrees[i]}')  # Mark with a red dot

    plt.legend()
    plt.grid(True)
    save_path = f'{save_dir}obj_precision_recall_{epoch}_{cloud_ratio}_legend.png'
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

    for branch_degree in branch_degrees:
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
    plt.title(f'Branch Precision-Recall Curve with at Epoch: {epoch}', fontsize=16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    save_path = f'{save_dir}branch_precision_recall_{epoch}_{cloud_ratio}.png'
    plt.savefig(save_path)
    plt.close()
    return save_path

def get_precision_and_recall(tp, fn, fp):
    """
    Calculates precision and recall given true positives, false negatives, and false positives.

    Args:
        tp (int): Number of true positives.
        fn (int): Number of false negatives.
        fp (int): Number of false positives.

    Returns:
        tuple: A tuple containing (precision, recall).
               Returns 0.0 for either if the denominator is zero to avoid division by zero errors.
    """
    precision = 0
    if (tp + fp > 0):
        precision = tp / (tp + fp)
    recall = 0
    if (tp + fn > 0):
        recall = tp / (tp + fn)
    return precision, recall

if __name__ == "__main__":
    ckpt_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/ckpt/"   

    run_id = 'AUT-330' # baseline
    epoch = 60
    # run_id = 'AUT-331' # no segclr
    # epoch = 40
    run_dir = f'{ckpt_dir}{run_id}/'
    
    ckpt_path = f'{run_dir}model_{epoch}.pt'
    with open(f'{run_dir}config.json', 'r') as f:
        config = json.load(f)
    if run_id == 'AUT-330' or run_id == 'AUT-331':
        config['data']['labels_dir'] = "labels_at_1300_ignore_inbetween/"
    
    config['loader']['num_workers'] = 32
    config['data']["data_dir"] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/"

    model = create_model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    mode = 'test'
    data = AutoProofDataset(config, mode)

    data_loader = build_dataloader(config, data, mode)

    model.eval()
    with torch.no_grad():
        root_to_output = {}

        obj_det_roots = data_utils.load_txt("/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/roots_343_1300/split_598963/test_conf_no_error_in_box_roots.txt")
        # num_roots_to_sample = int(len(obj_det_roots) * 0.10)
        # sampled_indices = np.random.choice(len(obj_det_roots), size=num_roots_to_sample, replace=False)
        # obj_det_roots = set(obj_det_roots[sampled_indices])
        
        with tqdm(total=len(data) / config['loader']['batch_size'], desc=mode) as pbar:
            for i, data in enumerate(data_loader):
                # root (b, 1) input (b, fov, d), labels (b, fov, 1), conf (b, fov, 1), adj (b, fov, fov)
                roots, input, labels, confidences, dist_to_error, rank, adj, _ = data
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
        ignored_roots = []
        other_roots = []
        with tqdm(total=(len(obj_det_data) / config['loader']['batch_size']), desc="obj det all") as pbar:
            for i, data in enumerate(obj_det_loader):
                roots, missed, found, ignored = data
                for i, root in enumerate(roots):
                    if missed[i]:
                        missed_roots.append(root)
                    elif found[i]:
                        found_roots.append(root)
                    elif ignored[i]:
                        ignored_roots.append(root)
                    else:
                        other_roots.append(root)
                
                pbar.update()
        end_time = time.time()

        save_dir = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/test/'
        data_utils.save_txt(f"{save_dir}missed.txt", missed_roots)
        data_utils.save_txt(f"{save_dir}found.txt", found_roots)
        data_utils.save_txt(f"{save_dir}other.txt", other_roots)
        data_utils.save_txt(f"{save_dir}ignored.txt", ignored_roots)
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