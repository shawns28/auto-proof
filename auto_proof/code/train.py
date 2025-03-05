from auto_proof.code.dataset import build_dataloader
from auto_proof.code.visualize import get_root_output, visualize

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
import math
import numpy as np
import neptune
import multiprocessing
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

class Trainer(object):
    def __init__(self, config, model, train_dataset, val_dataset, test_dataset, run):
        self.run = run
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run["gpu"] = torch.cuda.get_device_name(0)
        self.model = model.to(self.device)
        self.config = config
        self.ckpt_dir = config['trainer']['ckpt_dir']
        self.save_every = config['trainer']['save_ckpt_every']
        self.batch_size = config['loader']['batch_size']
        self.data_dir = config['data']['data_dir']

        ### datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = build_dataloader(config, train_dataset, run, 'train')
        self.val_loader = build_dataloader(config, val_dataset, run, 'val')
        self.test_loader = build_dataloader(config, test_dataset, run, 'test')
        self.train_size = len(train_dataset) 
        self.val_size = len(val_dataset)
        self.test_size = len(test_dataset)

        self.class_weights = torch.tensor([config['trainer']['class_weights'], 10 - config['trainer']['class_weights']]).float().to(self.device)

        ### trainings params
        self.epochs = config['optimizer']['epochs']
        self.epoch = 0
        self.lr = config['optimizer']['lr']

        self.visualize_rand_num = config['trainer']['visualize_rand_num']
        self.visualize_cutoff = config['trainer']['visualize_cutoff']
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=self.lr)
        # Should try the weiss paper schedular and CosineAnnealingLR and CosineAnnealingWarmRestarts
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1)

    def train(self):     
        best_vloss = 1_000_000
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print('Initial Epoch {}/{} | LR {:.4f}'.format(self.epoch, self.epochs, self.optimizer.state_dict()['param_groups'][0]['lr']))
        start_datetime = datetime.now()
        self.run["train/start"] = start_datetime

        save_dir = f'{self.ckpt_dir}{timestamp}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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
                avg_vloss = self.val_epoch(save_dir)
                self.run["val/avg_loss"].append(avg_vloss)

                print("Starting visualization")
                visualize_start = datetime.now()
                self.visualize_examples(save_dir)                    
                visualize_end = datetime.now()
                print("visualizing took", visualize_end - visualize_start)

            print('Epoch {}/{} | Loss {:.5f} | Val loss {:.5f} | LR {:.6f}'.format(self.epoch, self.epochs, avg_loss, avg_vloss, self.optimizer.state_dict()['param_groups'][0]['lr']))
            
            self.scheduler.step(avg_vloss)

            if avg_vloss < best_vloss and self.epoch >= (self.epochs // 2):
                best_vloss = avg_vloss
                self.save_checkpoint(save_dir)
            elif self.epoch % self.save_every == 0:
                self.save_checkpoint(save_dir)
            
        end_datetime = datetime.now()
        total_time = end_datetime - start_datetime
        self.run["total_time"] = total_time

    def train_epoch(self):
        running_loss = 0
        with tqdm(total=self.train_size / self.batch_size, desc="train") as pbar:
            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                input, labels, confidence, adj = [x.float().to(self.device) for x in data]
                # self.model.train() # Marking this here due to async

                output = self.model(input, adj)
                loss = self.model.compute_loss(output, labels, confidence, self.class_weights)
                # self.run["train/loss"].append(loss)

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
        

    def val_epoch(self, save_dir):
        # Thresholds for merge error
        running_vloss = 0

        recall_targets = [0.99, 0.95, 0.9, 0.5]

        total_merge_prob = torch.zeros(1).to(self.device)
        total_labels = torch.zeros(1).to(self.device)

        with tqdm(total=self.val_size / self.batch_size, desc="val") as pbar:
            for i, data in enumerate(self.val_loader):
                input, labels, confidence, adj = [x.float().to(self.device) for x in data]
                output = self.model(input, adj)
                loss = self.model.compute_loss(output, labels, confidence, self.class_weights)
                self.run["val/loss"].append(loss)
                running_vloss += loss

                sigmoid = nn.Sigmoid()
                output_prob = sigmoid(output)
                # The first column is for the merge error  probabilities
                output_merge_prob = output_prob[:, :, 0] # [:, :, 0] to account for batches
                # output = torch.argmax(output, dim=-1)
                labels = labels.squeeze(-1)

                mask = labels != -1
                output_merge_prob = output_merge_prob[mask]
                labels = labels[mask]

                total_merge_prob = torch.cat((total_merge_prob, output_merge_prob), dim=0)
                total_labels = torch.cat((total_labels, labels), dim=0)

                pbar.update()

        self.statistics(total_labels, total_merge_prob, recall_targets, save_dir)
        
        return running_vloss / (self.val_size / self.batch_size)

    def statistics(self, total_labels, total_merge_prob, recall_targets, save_dir):
        # Flipping the labels to make merge errors 1 for scikit-learn precision recall curve
        # Remove first element because that was for initialization
        total_labels = 1 - total_labels.cpu().detach().numpy()[1:]
        total_merge_prob = total_merge_prob.cpu().detach().numpy()[1:]

        precision_curve, recall_curve, threshold_curve = precision_recall_curve(total_labels, total_merge_prob)

        plt.figure(figsize=(8, 8))
        plt.plot(recall_curve, precision_curve, marker='.', markersize=2, label='Precision-Recall curve')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Merge Error Precision-Recall Curve')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        for target_recall in recall_targets:
            pinned_threshold, pinned_precision, pinned_recall = self.find_pinned_recall(precision_curve, recall_curve, threshold_curve, target_recall)
            plt.scatter(pinned_recall, pinned_precision, c='red', s=30, label=f'Target Recall: {target_recall:.2f}, Recall: {pinned_recall:.2f}, Precision: {pinned_precision:.2f}, Threshold: {pinned_threshold:.2f}')  # Mark with a red dot
        
        plt.legend()
        plt.grid(True)
        save_path = f'{save_dir}/precision_recall_{self.epoch}.png'
        plt.savefig(save_path)
        self.run["pinned/precision_recall_curve"].append(neptune.types.File(save_path))

    def find_pinned_recall(self, precision_curve, recall_curve, threshold_curve, target_recall):
        pinned_idx = np.argmin(np.abs(recall_curve - target_recall))
        # Invert this to be consistent with other visualization
        pinned_threshold = 1 - threshold_curve[pinned_idx]
        pinned_precision = precision_curve[pinned_idx]
        pinned_recall = recall_curve[pinned_idx]

        return pinned_threshold, pinned_precision, pinned_recall

    # Not using confidence for the accuracy currently
    def accuracy(self, output, labels):
        tp = torch.sum(torch.logical_and(output == 1, labels == 1)).item()
        tn = torch.sum(torch.logical_and(output == 0, labels == 0)).item()
        fp = torch.sum(torch.logical_and(output == 1, labels == 0)).item()
        fn = torch.sum(torch.logical_and(output == 0, labels == 1)).item()

        return tp, tn, fp, fn

    def save_checkpoint(self, save_dir):
        model_path = 'model_{}.pt'.format(self.epoch)
        complete_path = f'{save_dir}/{model_path}'
        torch.save(self.model.state_dict(), complete_path)
        self.run["model_ckpts"].upload(complete_path)

    def visualize_examples(self, save_dir):
        dataset_dict = {'train': self.train_dataset, 'val': self.val_dataset, 'test': self.test_dataset}

        visualize_tuples = [
            (864691135463333789, True, 'train'), 
            (864691136379030485, True, 'val'), 
            (864691136443843459, True, 'test'), 
            (864691134918370314, True, 'test')]
        
        for _ in range(self.visualize_rand_num):
            mode, dataset = random.choice(list(dataset_dict.items()))
            root = dataset.get_random_root()
            while dataset.get_num_initial_vertices(root) > self.visualize_cutoff:
                root = dataset.get_random_root()
            visualize_tuples.append((root, False, mode))

        with tqdm(total=len(visualize_tuples)) as pbar:
            for visualize_tuple in visualize_tuples:
                root, is_constant, mode = visualize_tuple
                self.visualize_root(root, is_constant, mode, dataset_dict, save_dir)
                pbar.update()
    
    def visualize_root(self, root, is_constant, mode, dataset_dict, save_dir):
        try:
            dataset = dataset_dict[mode]
            vertices, edges, labels, confidence, output, root_mesh, is_proofread, _ = get_root_output(self.model, self.device, dataset, root)
            random = 'random'
            if is_constant:
                random = 'constant'
            edit = 'edit'
            if is_proofread:
                edit = 'proofread'
            save_path = f'{save_dir}/{self.epoch}_{random}_{mode}_{edit}_{root}.html'
            visualize(vertices, edges, labels, confidence, output, root_mesh, save_path)
            self.run[f"visuals/{self.epoch}/{random}_{mode}_{edit}_{root}"].upload(save_path)
        except Exception as e:
            print("Failed visualization for root id: ", root, "error: ", e)