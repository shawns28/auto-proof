from auto_proof.code.dataset import build_dataloader
from auto_proof.code.visualize import get_root_output, visualize

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
import math
import numpy as np
import neptune
import multiprocessing


def visualize_examples(model, device, timestamp, train_dataset, val_dataset, test_dataset, visualize_cutoff, epoch, run, ckpt_dir, visualize_rand_num):
    # Change literally all of these examples because they're not pretty
    constant_root_tuples = [(864691135778235581, 'train'), (864691136379030485,  'val'), (864691136443843459, 'test'), (864691134918370314, 'test')]
    dataset_dict = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

    # device = torch.device("cpu")
    # model.to(device)
    # model.eval()

    save_dir = f'{ckpt_dir}{timestamp}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for constant_root_tuple in constant_root_tuples:
        visualize_root(model, device, constant_root_tuple[0], True, constant_root_tuple[1], dataset_dict, epoch, run, save_dir)
    for i in range(visualize_rand_num):
        for mode, dataset in dataset_dict.items():
            root = dataset.get_random_root()
            while dataset.get_num_initial_vertices(root) > visualize_cutoff:
                root = dataset.get_random_root()
            visualize_root(model, device, root, False, mode, dataset_dict, epoch, run, save_dir)

def visualize_root(model, device, root, is_constant, mode, dataset_dict, epoch, run, save_dir):
    #mprint("Visualize root", root)
    dataset = dataset_dict[mode]
    try:
        vertices, edges, labels, confidence, output, root_mesh, is_proofread, num_initial_vertices = get_root_output(model, device, dataset, root)
        random = 'random'
        if is_constant:
            random = 'constant'
        edit = 'edit'
        if is_proofread:
            edit = 'proofread'
        save_path = f'{save_dir}/{epoch}_{random}_{mode}_{edit}_{root}.html'
        visualize(vertices, edges, labels, confidence, output, root_mesh, save_path)
        run[f"visuals/{epoch}/{random}_{mode}_{edit}_{root}"].upload(save_path)
    except Exception as e:
        print("Failed visualization for root id: ", root, "error: ", e)

class Trainer(object):
    def __init__(self, config, model, train_dataset, val_dataset, test_dataset, run):
        self.run = run
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("gpu: ", torch.cuda.get_device_name(0))
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

        self.class_weights = torch.tensor([9, 1]).float().to(self.device)

        ### trainings params
        self.epochs = config['optimizer']['epochs']
        self.epoch = 0
        self.lr = config['optimizer']['lr']

        self.visualize_rand_num = config['trainer']['visualize_rand_num']
        self.visualize_cutoff = config['trainer']['visualize_cutoff']
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=self.lr)

    def train(self):     
        best_vloss = 1_000_000
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print('Initial Epoch {}/{} | LR {:.4f}'.format(self.epoch, self.epochs, self.optimizer.state_dict()['param_groups'][0]['lr']))
        start_datetime = datetime.now()
        self.run["train/start"] = start_datetime

        for i in range(self.epochs):
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
                # Will visualizing currently keep reseting between eval and non eval because of async
                # p = multiprocessing.Process(target=self.visualize_examples, args=(timestamp, constant_root_tuples, dataset_dict))
                # p.start()  # Start the visualization process in the background

                # Asynchronous Visualization (using a Pool for proper CUDA handling):
                # with multiprocessing.Pool(processes=1) as pool: # Create a process pool
                #     self.run["test"] = "hi"
                #     pool.apply_async(visualize_examples, (self.model, timestamp, self.train_dataset, self.val_dataset, self.test_dataset, self.visualize_cutoff, self.epoch, self.run, self.ckpt_dir, self.visualize_rand_num))
                #     pool.close()  # Important: Close the pool before joining
                #     pool.join()
                # visualize_examples(self.model, timestamp, self.train_dataset, self.val_dataset, self.test_dataset, self.visualize_cutoff, self.epoch, self.run, self.ckpt_dir, self.visualize_rand_num)
                visualize_start = datetime.now()
                visualize_examples(self.model, self.device, timestamp, self.train_dataset, self.val_dataset, self.test_dataset, self.visualize_cutoff, self.epoch, self.run, self.ckpt_dir, self.visualize_rand_num)                    
                visualize_end = datetime.now()
                print("visualizing took", visualize_end - visualize_start)

            print('Epoch {}/{} | Loss {:.5f} | Val loss {:.5f} | LR {:.6f}'.format(self.epoch + 1, self.epochs, avg_loss, avg_vloss, self.optimizer.state_dict()['param_groups'][0]['lr']))

            if avg_vloss < best_vloss and self.epoch >= (self.epochs // 2):
                best_vloss = avg_vloss
                self.save_checkpoint(timestamp)
            elif self.epoch % self.save_every == 0:
                self.save_checkpoint(timestamp)
        
        end_datetime = datetime.now()
        total_time = end_datetime - start_datetime
        self.run["total_time"] = total_time

    def train_epoch(self):
        running_loss = 0
        with tqdm(total=self.train_size / self.batch_size, desc="train") as pbar:
            for i, data in enumerate(self.train_loader):
                input, labels, confidence, adj, _ = [x.float().to(self.device) for x in data]
                # self.model.train() # Marking this here due to async

                self.optimizer.zero_grad()

                output = self.model(input, adj)
                loss = self.model.compute_loss(output, labels, confidence, self.class_weights)
                self.run["train/loss"].append(loss)

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
            # self.scheduler.step()
        return running_loss / (self.train_size / self.batch_size)
        

    def val_epoch(self):
        # Thresholds for merge error
        thresholds = [0.05, 0.1, 0.5, 0.9, 0.95]
        metrics = {}

        for threshold in thresholds:
            metrics[threshold] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        running_vloss = 0
    
        with tqdm(total=self.val_size / self.batch_size, desc="val") as pbar:
            for i, data in enumerate(self.val_loader):
                input, labels, confidence, adj, _ = [x.float().to(self.device) for x in data]
                output = self.model(input, adj)
                loss = self.model.compute_loss(output, labels, confidence, self.class_weights)
                self.run["val/loss"].append(loss)
                running_vloss += loss

                sigmoid = nn.Sigmoid()
                output_prob = sigmoid(output)
                # The first column is for the merge error probabilities
                output_merge_prob = output_prob[:, :, 0] # [:, :, 0] to account for batches
                # output = torch.argmax(output, dim=-1)
                labels = labels.squeeze(-1)

                for threshold in thresholds:
                    # Doing the inverse here since if we have 0.9 for merge error probability and a threshold of 0.7
                    # then we want the label for that to be 0. If the proabability of merge is higher than threshold
                    # then we want the label to be 0. 
                    thresholded_output = (output_merge_prob < threshold).int()

                    curr_tp, curr_tn, curr_fp, curr_fn = self.accuracy(thresholded_output, labels)
                    metrics[threshold]["tp"] += curr_tp
                    metrics[threshold]["tn"] += curr_tn
                    metrics[threshold]["fp"] += curr_fp
                    metrics[threshold]["fn"] += curr_fn
    
                pbar.update()
            
        # precision and recall for negative since that represents merge spots
        for threshold in thresholds:
            precision = 0
            recall = 0
            f1 = 0
            g_mean = 0

            # print("threshold: ", threshold)
            # print("tp/tn/fp/fn", metrics[threshold]["tp"], metrics[threshold]["tn"], metrics[threshold]["fp"], metrics[threshold]["fn"])

            if metrics[threshold]["tn"] + metrics[threshold]["fn"] != 0:
                precision = metrics[threshold]["tn"] / (metrics[threshold]["tn"] + metrics[threshold]["fn"])
            if metrics[threshold]["tn"] + metrics[threshold]["fp"] != 0:
                recall = metrics[threshold]["tn"] / (metrics[threshold]["tn"] + metrics[threshold]["fp"])
            if precision + recall != 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            if metrics[threshold]["tp"] + metrics[threshold]["fn"] != 0:
                true_recall = metrics[threshold]["tp"] / (metrics[threshold]["tp"] + metrics[threshold]["fn"])
                g_mean = math.sqrt(recall * true_recall)

            self.run[f"{threshold}/precision"].append(precision)
            self.run[f"{threshold}/recall"].append(recall)
            self.run[f"{threshold}/f1"].append(f1)
            self.run[f"{threshold}/g_mean"].append(g_mean)

        return running_vloss / (self.val_size / self.batch_size)

    # Not using confidence for the accuracy currently
    def accuracy(self, output, labels):
        mask = labels != -1
        output = output[mask]
        labels = labels[mask]

        tp = torch.sum(torch.logical_and(output == 1, labels == 1)).item()
        tn = torch.sum(torch.logical_and(output == 0, labels == 0)).item()
        fp = torch.sum(torch.logical_and(output == 1, labels == 0)).item()
        fn = torch.sum(torch.logical_and(output == 0, labels == 1)).item()

        return tp, tn, fp, fn

    def save_checkpoint(self, timestamp):
        directory_path = f'{self.ckpt_dir}/{timestamp}'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        model_path = 'model_{}.pt'.format(self.epoch)
        torch.save(self.model.state_dict(), f'{directory_path}/{model_path}')
        self.run["model_ckpts"].upload(f'{directory_path}/{model_path}')
