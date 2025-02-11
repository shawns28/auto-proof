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

        # Change literally all of these examples because they're not pretty
        constant_root_tuples = [(864691135778235581, 'train'), (864691136379030485,  'val'), (864691136443843459, 'test'), (864691134918370314, 'test')]
        dataset_dict = {'train': self.train_dataset, 'val': self.val_dataset, 'test': self.test_dataset}


        for i in range(self.epochs):
            self.epoch = i
            self.run["epoch"].append(self.epoch)
            self.run["lr"].append(self.optimizer.state_dict()['param_groups'][0]['lr'])

            # Run one epoch.
            self.model.train()
            avg_loss = self.train_epoch()
            self.run["train/avg_loss"] = avg_loss
            train_end_datetime = datetime.now()
            self.run["train/end"] = train_end_datetime

            self.model.eval()
            with torch.no_grad():
                avg_vloss, f1, recall, g_mean = self.val_epoch()
                self.run["val/avg_loss"].append(avg_vloss)
                self.run["val/f1"].append(f1)
                self.run["val/recall"].append(recall)
                self.run["g_mean"].append(g_mean)
                # Will visualizing currently keep reseting between eval and non eval because of async
                # p = multiprocessing.Process(target=self.visualize_examples, args=(timestamp, constant_root_tuples, dataset_dict))
                # p.start()  # Start the visualization process in the background

                # Asynchronous Visualization (using a Pool for proper CUDA handling):
                # with multiprocessing.Pool(processes=1) as pool: # Create a process pool
                #     pool.apply_async(self.visualize_examples, (timestamp, constant_root_tuples, dataset_dict)) # Corrected: Use apply_async

            print('Epoch {}/{} | Loss {:.5f} | Val loss {:.5f} | F1 {:.5f} | Recall {:.5f} | G-mean {:.5f}| LR {:.6f}'.format(self.epoch + 1, self.epochs, avg_loss, avg_vloss, f1, recall, g_mean, self.optimizer.state_dict()['param_groups'][0]['lr']))

            if avg_vloss < best_vloss and self.epoch >= (self.epochs // 2):
                best_vloss = avg_vloss
                self.save_checkpoint(timestamp)
            elif self.epoch % self.save_every == 0:
                self.save_checkpoint(timestamp)

            # visualize
            # visualize_start = datetime.now()
            # self.visualize_examples(timestamp, constant_root_tuples, dataset_dict)
            # visualize_end = datetime.now();
            # print("visualizing took", visualize_end - visualize_start)
        
        end_datetime = datetime.now()
        total_time = end_datetime - start_datetime
        self.run["total_time"] = total_time

    def visualize_examples(self,timestamp, constant_root_tuples, dataset_dict):
        save_dir = f'{self.ckpt_dir}{timestamp}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for constant_root_tuple in constant_root_tuples:
            self.visualize_root(constant_root_tuple[0], True, constant_root_tuple[1], dataset_dict, save_dir)
        for i in range(self.visualize_rand_num):
            root = self.train_dataset.get_random_root()
            while self.train_dataset.get_num_initial_vertices(root) > self.visualize_cutoff:
                root = self.train_dataset.get_random_root()
            self.visualize_root(root, False, 'train', dataset_dict, save_dir)

            root = self.val_dataset.get_random_root()
            while self.val_dataset.get_num_initial_vertices(root) > self.visualize_cutoff:
                root = self.train_dataset.get_random_root()
            self.visualize_root(root, False, 'val', dataset_dict, save_dir)

            root = self.test_dataset.get_random_root()
            while self.test_dataset.get_num_initial_vertices(root) > self.visualize_cutoff:
                root = self.test_dataset.get_random_root()
            self.visualize_root(root, False, 'test', dataset_dict, save_dir)

    def visualize_root(self, root, is_constant, mode, dataset_dict, save_dir):
        print("visualizing root", root)
        dataset = dataset_dict[mode]
        vertices, edges, labels, confidence, output, root_mesh, is_proofread, num_initial_vertices = get_root_output(self.model, "cpu", dataset, root)
        random = 'random'
        if is_constant:
            random = 'constant'
        edit = 'edit'
        if is_proofread:
            edit = 'proofread'
        save_path = f'{save_dir}/{self.epoch}_{random}_{mode}_{edit}_{root}.html'
        visualize(vertices, edges, labels, confidence, output, root_mesh, save_path)
        self.run[f"visuals/{self.epoch}/{random}_{mode}_{edit}_{root}"].upload(save_path)

    def train_epoch(self):
        running_loss = 0.
        last_loss = 0.
        with tqdm(total=self.train_size / self.batch_size, desc="train") as pbar:
            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)

                input, labels, confidence, adj, _ = [x.float().to(self.device) for x in data]
                self.model.train() # Marking this here due to async
                output = self.model(input, adj)
                loss = self.model.compute_loss(output, labels, confidence, self.class_weights)
                self.run["train/loss"].append(loss)

                # optimize 
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                # if self.epoch == 0:
                #     if i < 100:
                #         if i % 10 == 0:
                #             print('  batch {} loss: {}'.format(i + 1, last_loss))
                #     if i < 10:
                #         last_loss = running_loss / (i + 1)
                #         print('  batch {} loss: {}'.format(i + 1, last_loss))

                # if i % 1000 == 999:
                #     last_loss = running_loss / (i + 1)
                #     print('  batch {} loss: {}'.format(i + 1, last_loss))
                pbar.update()
            # self.scheduler.step()
            return running_loss / (i + 1)
        

    def val_epoch(self):
        running_vloss = 0.
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        with tqdm(total=self.val_size / self.batch_size, desc="val") as pbar:
            for i, data in enumerate(self.val_loader):
                input, labels, confidence, adj, _ = [x.float().to(self.device) for x in data]
                output = self.model(input, adj)
                loss = self.model.compute_loss(output, labels, confidence, self.class_weights)
                self.run["val/loss"].append(loss)
                running_vloss += loss
                sigmoid = nn.Sigmoid()
                output = sigmoid(output)
                output = torch.argmax(output, dim=-1)
                curr_tp, curr_tn, curr_fp, curr_fn = self.model.accuracy(output, labels)
                tp += curr_tp
                tn += curr_tn
                fp += curr_fp
                fn += curr_fn
                pbar.update()
            
            # precision and recall for negative since that represents merge spots
            precision = 0
            recall = 0
            f1 = 0
            if tn + fn != 0:
                precision = tn / (tn + fn)
            if tn + fp != 0:
                recall = tn / (tn + fp)
            if precision + recall != 0:
                f1 = 2 * (precision * recall) / (precision + recall)


            # g-mean
            true_recall = tp / (tp + fn)
            g_mean = math.sqrt(recall * true_recall)

            return running_vloss / (i + 1), f1, recall, g_mean


    def save_checkpoint(self, timestamp):
        directory_path = f'{self.ckpt_dir}/{timestamp}'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        model_path = 'model_{}.pt'.format(self.epoch)
        torch.save(self.model.state_dict(), f'{directory_path}/{model_path}')
        self.run["model_ckpts"].upload(f'{directory_path}/{model_path}')
