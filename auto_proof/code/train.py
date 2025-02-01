from auto_proof.code.dataset import build_dataloader, AutoProofDataset
from auto_proof.code.visualize import get_root_output, visualize
from auto_proof.code.pre.data_utils import initialize

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

class Trainer(object):
    def __init__(self, config, model, dataset, run):
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
        _, _, _, client, _ = initialize()
        self.client = client

        ### datasets
        train_loader, val_loader, train_dataset, val_dataset = build_dataloader(dataset, config, run)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_size = len(train_dataset) 
        self.val_size = len(val_dataset)

        self.class_weights = torch.tensor([9, 1]).float().to(self.device)

        ### trainings params
        self.epochs = config['optimizer']['epochs']
        self.epoch = 0
        self.lr = config['optimizer']['lr']
        self.max_iter = config['optimizer']['max_iterations']
        self.curr_iter = 0
        self.visualize_num = config['trainer']['visualize_num']
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=self.lr)
        # lambda1 = lambda epoch: epoch / self.epochs
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.warmup_cosine_decay(epoch, self.epochs // 10, self.epochs))
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1)

    # Find better one
    def warmup_cosine_decay(self, epoch, warmup_epochs, total_epochs):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    def train(self):     
        best_vloss = 1_000_000
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print('Initial Epoch {}/{} | LR {:.4f}'.format(self.epoch, self.epochs, self.optimizer.state_dict()['param_groups'][0]['lr']))
        start_datetime = datetime.now()
        self.run["train/start"] = start_datetime
        # with tqdm(total=self.epochs) as tqdm_tracker:
        while self.curr_iter < self.max_iter:
            self.run["epoch"].append(self.epoch)
            self.run["lr"].append(self.optimizer.state_dict()['param_groups'][0]['lr'])

            # Run one epoch.
            self.model.train()
            avg_loss = self.train_epoch()
            self.run["train/avg_loss"] = avg_loss
            train_end_datetime = datetime.now()
            self.run["train/end"] = train_end_datetime
    
            with torch.no_grad():
                self.model.eval()
                avg_vloss, f1, recall, g_mean = self.val_epoch()
                self.run["val/avg_loss"].append(avg_vloss)
                self.run["val/f1"].append(f1)
                self.run["val/recall"].append(recall)
                self.run["g_mean"].append(g_mean)

            self.scheduler.step(avg_vloss)
            print('Epoch {}/{} | Loss {:.5f} | Val loss {:.5f} | F1 {:.5f} | Recall {:.5f} | G-mean {:.5f}| LR {:.6f}'.format(self.epoch, self.epochs, avg_loss, avg_vloss, f1, recall, g_mean, self.optimizer.state_dict()['param_groups'][0]['lr']))

            if avg_vloss < best_vloss and self.epoch >= (self.epochs // 2):
                best_vloss = avg_vloss
                self.save_checkpoint(timestamp)
            elif self.curr_iter % self.save_every == 0:
                self.save_checkpoint(timestamp)

            # visualize
            save_dir = f'{self.ckpt_dir}{timestamp}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i in range(self.visualize_num):
                rand_idx = np.random.randint(0, self.val_size)
                vertices, edges, labels, confidence, output, root = get_root_output(self.model, self.device, self.val_dataset, rand_idx, self.client)
                save_path = f'{save_dir}/{root}_{self.epoch}.html'
                visualize(vertices, edges, labels, confidence, output, save_path)
                self.run[f"visuals/{self.epoch}/{root}"].upload(save_path)
            self.epoch += 1
        end_datetime = datetime.now()
        total_time = end_datetime - start_datetime
        self.run["total_time"] = total_time


    def train_epoch(self):
        running_loss = 0.
        last_loss = 0.
        with tqdm(total=self.train_size / self.batch_size, desc="train") as pbar:
            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)

                input, labels, confidence, adj, _ = [x.float().to(self.device) for x in data]

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
                self.curr_iter += 1
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
