from auto_proof.code.dataset import build_dataloader
from auto_proof.code.visualize import visualize, get_root_output
from auto_proof.code.pre import data_utils
from auto_proof.code.object_detection import ObjectDetectionDataset, obj_det_dataloader, obj_det_plot_legend

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
import numpy as np
import neptune
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import json
from prodigyopt import Prodigy

class Trainer(object):
    """
    Manages the training, validation, and evaluation lifecycle of the AutoProof model.

    This class handles data loading, model optimization, loss computation,
    checkpointing, visualization, and metric logging (including object detection
    and node-based precision-recall curves).
    """
    def __init__(self, config, model, train_dataset, val_dataset, test_dataset, run):
        """
        Initializes the Trainer.

        Args:
            config (dict): A dictionary containing all configuration parameters
                           for the trainer, loader, data, and model.
            model (nn.Module): The model to be trained.
            train_dataset (torch.utils.data.Dataset): The dataset for training.
            val_dataset (torch.utils.data.Dataset): The dataset for validation.
            test_dataset (torch.utils.data.Dataset): The dataset for testing.
            run (neptune.init_run): The Neptune AI experiment run object for logging.
        """
        self.run = run
        self.run_id = run["sys/id"].fetch()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run["gpu"] = torch.cuda.get_device_name(0)
        self.model = model.to(self.device)
        self.config = config

        self.data_dir = config['data']['data_dir']
        self.box_cutoff = config['data']['box_cutoff']
        mat_version_start = config['data']['mat_version_start']
        mat_version_end = config['data']['mat_version_end']
        self.roots_dir = f'{self.data_dir}roots_{mat_version_start}_{mat_version_end}/'
        self.split_dir = f'{self.roots_dir}{config['data']['split_dir']}'

        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = build_dataloader(config, train_dataset, 'train')
        self.val_loader = build_dataloader(config, val_dataset, 'val')
        self.test_loader = build_dataloader(config, test_dataset, 'test')
        self.train_size = len(train_dataset) 
        self.val_size = len(val_dataset)
        self.test_size = len(test_dataset)

        self.batch_size = config['loader']['batch_size']

        self.ckpt_dir = f'{self.data_dir}{config['trainer']['ckpt_dir']}'
        self.save_dir = f'{self.ckpt_dir}{self.run_id}/'
        self.save_ckpt_every = config['trainer']['save_ckpt_every']
        self.save_visual_every = config['trainer']['save_visual_every']
        self.show_tol = config['trainer']['show_tol']
        self.obj_det_error_cloud_ratios = config['trainer']['obj_det_error_cloud_ratios']
        self.max_dist = config['trainer']['max_dist']
        self.visualize_rand_num = config['trainer']['visualize_rand_num']
        self.visualize_cutoff = config['trainer']['visualize_cutoff']
        self.tresholds = self.config['trainer']['thresholds']
        self.recall_targets = self.config['trainer']['recall_targets']
        self.branch_degrees = self.config['trainer']['branch_degrees']

        # Loss weights
        self.class_weights = torch.tensor(config['trainer']['class_weights']).float().to(self.device)
        self.conf_weight = config['trainer']['conf_weight']
        self.tolerance_weight = config['trainer']['tolerance_weight']
        self.box_weight = config['trainer']['box_weight']

        self.obj_det_val_roots = set(data_utils.load_txt(f'{self.split_dir}{config['data']['obj_det_val_roots']}'))

        ### trainings params
        self.epochs = config['optimizer']['epochs']
        self.epoch = 0
        self.lr = config['optimizer']['lr']
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1)
        
    def train(self):
        """
        Executes the main training loop.

        This method iterates through epochs, performs training and validation,
        logs metrics to Neptune, saves checkpoints, and generates visualizations.
        """     
        best_vloss = 1_000_000
        print('Initial Epoch {}/{} | LR {:.4f}'.format(self.epoch, self.epochs, self.optimizer.state_dict()['param_groups'][0]['lr']))
        start_datetime = datetime.now()
        self.run["train/start"] = start_datetime

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        new_config_path = f'{self.save_dir}config.json'
        with open(new_config_path, 'w') as file:
            json.dump(self.config, file)

        for i in range(1, self.epochs + 1):
            self.epoch = i
            self.run["epoch"].append(self.epoch)
            self.run["lr"].append(self.optimizer.state_dict()['param_groups'][0]['lr'])

            # Run one epoch.
            self.model.train()
            avg_loss = self.train_epoch()
            self.run["train/avg_loss"].append(avg_loss)
            train_end_datetime = datetime.now()
            self.run["train/end"] = train_end_datetime

            self.model.eval()
            with torch.no_grad():
                avg_vloss = self.val_epoch()
                self.run["val/avg_loss"].append(avg_vloss)

                if self.epoch <= 5 or self.epoch % self.save_visual_every == 0:
                    print("Starting visualization")
                    visualize_start = datetime.now()
                    self.visualize_examples()                    
                    visualize_end = datetime.now()
                    print("visualizing took", visualize_end - visualize_start)

            print('Epoch {}/{} | Loss {:.5f} | Val loss {:.5f} | LR {:.6f}'.format(self.epoch, self.epochs, avg_loss, avg_vloss, self.optimizer.state_dict()['param_groups'][0]['lr']))
            
            self.scheduler.step(avg_vloss)

            if avg_vloss < best_vloss and self.epoch >= (self.epochs // 2):
                best_vloss = avg_vloss
                self.save_checkpoint()
            elif self.epoch % self.save_ckpt_every == 0:
                self.save_checkpoint()
            
        end_datetime = datetime.now()
        total_time = end_datetime - start_datetime
        self.run["total_time"] = total_time

    def train_epoch(self):
        """
        Runs a single training epoch.

        Iterates through the training DataLoader, performs forward pass, computes loss,
        backpropagates, and updates model weights. Logs batch-level losses in early stages.

        Returns:
            float: The average training loss for the epoch.
        """
        running_loss = 0
        with tqdm(total=self.train_size / self.batch_size, desc="train") as pbar:
            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                input, labels, confidences, dist_to_error, rank, adj = [data[i].float().to(self.device) for i in range(1, len(data) - 1)]

                output = self.model(input, adj)
                loss = self.model.compute_loss(output, labels, confidences, dist_to_error, self.max_dist, self.class_weights, self.conf_weight, self.tolerance_weight, rank, self.box_cutoff, self.box_weight)

                # optimize 
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if self.epoch == 0:
                    if i < 1000:
                        if i % 100 == 0:
                            self.run["train/loss"].append(running_loss / (i + 1))
                    if i < 100:
                        if i % 10 == 0:
                            self.run["train/loss"].append(running_loss / (i + 1))
                    if i < 10:
                        self.run["train/loss"].append(running_loss / (i + 1))
                pbar.update()
        return running_loss / (self.train_size / self.batch_size)
        

    def val_epoch(self):
        """
        Runs a single validation epoch.

        Iterates through the validation DataLoader, performs forward pass, computes loss,
        and collects data for various metric evaluations (node-based PR curves,
        object detection metrics). Logs validation losses and plots to Neptune.

        Returns:
            float: The average validation loss for the epoch.
        """
        running_vloss = 0

        total_output = torch.zeros(1).to(self.device)
        total_labels = torch.zeros(1).to(self.device)
        total_output_conf = torch.zeros(1).to(self.device)
        total_labels_conf = torch.zeros(1).to(self.device)

        root_to_output = {}

        with tqdm(total=self.val_size / self.batch_size, desc="val") as pbar:
            for i, data in enumerate(self.val_loader):
                # root (b, 1) input (b, fov, d), labels (b, fov, 1), conf (b, fov, 1), adj (b, fov, fov)
                roots = data[0]
                input, labels, confidences, dist_to_error, rank, adj= [data[i].float().to(self.device) for i in range(1, len(data) - 1)]
                output = self.model(input, adj) # (b, fov, 1)
                loss = self.model.compute_loss(output, labels, confidences, dist_to_error, self.max_dist, self.class_weights, self.conf_weight, self.tolerance_weight,  rank, self.box_cutoff, self.box_weight)
                self.run["val/loss"].append(loss)
                running_vloss += loss

                sigmoid = nn.Sigmoid()
                output = sigmoid(output).squeeze(-1) # (b, fov)
                labels = labels.squeeze(-1) # (b, fov)
                confidences = confidences.squeeze(-1) # (b, fov)
                dist_to_error = dist_to_error.squeeze(-1) # (b, fov)

                # Used for all node based evaluation
                mask = labels != -1
                output_masked = output[mask] # (b * fov - buffer)
                labels_masked = labels[mask] # (b * fov - buffer)
                dist_to_error_masked = dist_to_error[mask] # (b * fov - buffer)
                dist_mask = torch.logical_or(dist_to_error_masked == 0, dist_to_error_masked > self.max_dist) # (b * fov - buffer)
                output_masked = output_masked[dist_mask] # (b * fov - buffer - tolerance)
                labels_masked = labels_masked[dist_mask] # (b * fov - buffer - tolerance)
                total_output = torch.cat((total_output, output_masked), dim=0)
                total_labels = torch.cat((total_labels, labels_masked), dim=0)

                # Used for confident only node based evaluation
                conf_mask = confidences == True
                output_conf = output[conf_mask]
                labels_conf = labels[conf_mask]
                dist_to_error_masked = dist_to_error[conf_mask]
                dist_mask = torch.logical_or(dist_to_error_masked == 0, dist_to_error_masked > self.max_dist)
                output_conf = output_conf[dist_mask]
                labels_conf = labels_conf[dist_mask]
                total_output_conf = torch.cat((total_output_conf, output_conf), dim=0)
                total_labels_conf = torch.cat((total_labels_conf, labels_conf), dim=0)

                # Used for object detection (connected component) evaluation
                output = output.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()
                if self.epoch <= 5 or self.epoch % self.save_visual_every == 0:
                    for i in range(len(roots)):
                        if roots[i] in self.obj_det_val_roots:
                            root_to_output[roots[i]] = output[i][mask[i]]

                pbar.update()

        # Object detection (connected component) evaluation
        if self.epoch <= 5 or self.epoch % self.save_visual_every == 0:
            print("creating obj det dataset")
            obj_det_data = ObjectDetectionDataset(self.config, root_to_output)
            obj_det_loader_all = obj_det_dataloader(self.config, obj_det_data)

            with tqdm(total=len(obj_det_data) / self.batch_size, desc="obj det") as pbar:
                for i, data in enumerate(obj_det_loader_all):
                    pbar.update()
            
            metrics_dict = obj_det_data.__getmetrics__()

            for cloud_ratio in self.obj_det_error_cloud_ratios:
                obj_save_path = obj_det_plot_legend(metrics_dict, self.tresholds, self.epoch, self.save_dir, cloud_ratio, self.box_cutoff, self.branch_degrees)
                self.run[f"plots/obj_precision_recall_curve_{cloud_ratio}"].append(neptune.types.File(obj_save_path))

        # NOTE: Theses plots are not useful other than being a sanity check
        # Node based evaluation
        total_labels = total_labels.cpu().detach().numpy()[1:]
        total_output = total_output.cpu().detach().numpy()[1:]
        self.pinned_plot(total_labels, total_output, self.recall_targets, 'all')

        total_labels_conf = total_labels_conf.cpu().detach().numpy()[1:]
        total_output_conf = total_output_conf.cpu().detach().numpy()[1:]
        self.pinned_plot(total_labels_conf, total_output_conf, self.recall_targets, 'confident')
        
        return running_vloss / (self.val_size / self.batch_size)

    def pinned_plot(self, total_labels, total_output, recall_targets, mode):
        """
        Generates and logs a precision-recall curve with specific recall targets highlighted.

        This plot shows the trade-off between precision and recall for node-level predictions.
        Points corresponding to predefined `recall_targets` are marked and their
        associated precision and threshold are displayed.

        Args:
            total_labels (np.ndarray): Array of true labels (binary).
            total_output (np.ndarray): Array of model predicted probabilities.
            recall_targets (list of float): A list of recall values to pinpoint on the curve.
            mode (str): A string indicating the plot mode (e.g., 'all' for all nodes, 'confident' for confident nodes).
        """

        precision_curve, recall_curve, threshold_curve = precision_recall_curve(total_labels, total_output)

        plt.figure(figsize=(8, 8))
        plt.plot(recall_curve, precision_curve, marker='.', markersize=2, label='Precision-Recall curve')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        if mode == 'all':
            plt.title('Merge Error Precision-Recall Curve for All Nodes With Tolerance')
        else:
            plt.title('Merge Error Precision-Recall Curve for Confident Nodes With Tolerance')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        for target_recall in recall_targets:
            pinned_threshold, pinned_precision, pinned_recall = self.find_pinned_recall(precision_curve, recall_curve, threshold_curve, target_recall)
            plt.scatter(pinned_recall, pinned_precision, c='red', s=30, label=f'Target Recall: {target_recall:.2f}, Recall: {pinned_recall:.2f}, Precision: {pinned_precision:.2f}, Threshold: {pinned_threshold:.2f}')  # Mark with a red dot
        
        plt.legend()
        plt.grid(True)
        save_path = f'{self.save_dir}node_precision_recall_{mode}_{self.epoch}.png'
        plt.savefig(save_path)
        self.run[f"plots/node_{mode}_precision_recall_curve"].append(neptune.types.File(save_path))

    def find_pinned_recall(self, precision_curve, recall_curve, threshold_curve, target_recall):
        """
        Finds the precision, recall, and threshold closest to a target recall value on a PR curve.

        Args:
            precision_curve (np.ndarray): Array of precision values from `precision_recall_curve`.
            recall_curve (np.ndarray): Array of recall values from `precision_recall_curve`.
            threshold_curve (np.ndarray): Array of threshold values from `precision_recall_curve`.
            target_recall (float): The desired recall value to pinpoint.

        Returns:
            tuple: A tuple containing (pinned_threshold, pinned_precision, pinned_recall).
        """
        pinned_idx = np.argmin(np.abs(recall_curve - target_recall))
        pinned_threshold = threshold_curve[pinned_idx]
        pinned_precision = precision_curve[pinned_idx]
        pinned_recall = recall_curve[pinned_idx]

        return pinned_threshold, pinned_precision, pinned_recall

    def save_checkpoint(self):
        """
        Saves the current state of the model as a checkpoint.

        The checkpoint is named by the current epoch and uploaded to Neptune.
        """
        model_path = 'model_{}.pt'.format(self.epoch)
        complete_path = f'{self.save_dir}{model_path}'
        torch.save(self.model.state_dict(), complete_path)
        self.run["model_ckpts"].upload(complete_path)

    def visualize_examples(self):
        """
        Generates and uploads visualizations of model predictions for selected examples.

        Includes a set of constant roots for consistent tracking and a number of random roots.
        Visualizations are saved as HTML files and uploaded to Neptune.
        """
        dataset_dict = {'train': self.train_dataset, 'val': self.val_dataset, 'test': self.test_dataset}

        visualize_tuples = [
            ('864691135657412579_000', True, 'val'), 
            ('864691135682085268_000', True, 'val'), 
            ('864691135884404592_000', True, 'val'), 
            ('864691135187356435_000', True, 'val')]
        
        for _ in range(self.visualize_rand_num):
            mode, dataset = random.choice(list(dataset_dict.items()))
            root = dataset.get_random_root()
            while dataset.get_num_initial_vertices(root) > self.visualize_cutoff:
                root = dataset.get_random_root()
            visualize_tuples.append((root, False, mode))

        with tqdm(total=len(visualize_tuples)) as pbar:
            for visualize_tuple in visualize_tuples:
                root, is_constant, mode = visualize_tuple
                self.visualize_root(root, is_constant, mode, dataset_dict)
                pbar.update()
    
    def visualize_root(self, root, is_constant, mode, dataset_dict):
        """
        Generates and uploads a single visualization for a specified root.

        This function calls the `visualize` utility to create an interactive HTML
        visualization of the neuron skeleton, ground truth labels, and model predictions.
        The visualization is saved locally and then uploaded to Neptune.

        Args:
            root (str): The ID of the root to visualize.
            is_constant (bool): True if the root is from the predefined constant set, False if random.
            mode (str): The dataset mode ('train', 'val', or 'test') the root belongs to.
            dataset_dict (dict): A dictionary mapping mode strings to their respective datasets.
        """
        try:
            dataset = dataset_dict[mode]
            vertices, edges, labels, confidences, output, root_mesh, is_proofread, _, dist_to_error, _, _, _, rank, _ = get_root_output(self.model, self.device, dataset, root)
            random = 'random'
            if is_constant:
                random = 'constant'
            edit = 'edit'
            if is_proofread:
                edit = 'proofread'
            save_path = f'{self.save_dir}{self.epoch}_{random}_{mode}_{edit}_{root}.html'
            visualize(vertices, edges, labels, confidences, output, root_mesh, dist_to_error, self.max_dist, self.show_tol, rank, self.box_cutoff, save_path)
            self.run[f"visuals/{self.epoch}/{random}_{mode}_{edit}_{root}"].upload(save_path)
        except Exception as e:
            print("Failed visualization for root id: ", root, "error: ", e)