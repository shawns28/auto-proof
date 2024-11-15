import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
import math

class Trainer(object):
    def __init__(self, config, model, dataloaders, data_sizes):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("gpu: ", torch.cuda.get_device_name(0))
        
        self.model = model.to(self.device)
        self.config = config
        self.ckpt_dir = config['trainer']['ckpt_dir']
        self.save_every = config['trainer']['save_ckpt_every']
        self.batch_size = config['loader']['batch_size']  

        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader = dataloaders[1]
        self.train_size = data_sizes[0]
        self.val_size = data_sizes[1]

        self.class_weights = torch.tensor([9, 1]).float().to(self.device)

        ### trainings params
        self.epochs = config['optimizer']['epochs']
        self.epoch = 0
        self.lr = config['optimizer']['lr']
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
        # with tqdm(total=self.epochs) as tqdm_tracker:
        for epoch in range(self.epochs):
            self.epoch = epoch
            # Run one epoch.
            self.model.train()
            avg_loss = self.train_epoch()
            
            with torch.no_grad():
                self.model.eval()
                avg_vloss, f1, recall, g_mean = self.val_epoch()

            self.scheduler.step(avg_vloss)
            print('Epoch {}/{} | Loss {:.5f} | Val loss {:.5f} | F1 {:.5f} | Recall {:.5f} | G-mean {:.5f}| LR {:.6f}'.format(epoch, self.epochs, avg_loss, avg_vloss, f1, recall, g_mean, self.optimizer.state_dict()['param_groups'][0]['lr']))

            if avg_vloss < best_vloss and epoch >= (self.epochs // 2):
                best_vloss = avg_vloss
                self.save_checkpoint(epoch, timestamp)
            elif epoch % self.save_every == 0:
                self.save_checkpoint(epoch, timestamp)

            epoch += 1


    def train_epoch(self):
        running_loss = 0.
        last_loss = 0.
        with tqdm(total=self.train_size / self.batch_size, desc="train") as pbar:
            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)

                input, labels, confidence, adj = [x.float().to(self.device) for x in data]

                output = self.model(input, adj)
                loss = self.model.compute_loss(output, labels, confidence, self.class_weights)

                # optimize 
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if self.epoch == 0:
                    if i < 100:
                        if i % 10 == 0:
                            print('  batch {} loss: {}'.format(i + 1, last_loss))
                    if i < 10:
                        last_loss = running_loss / (i + 1)
                        print('  batch {} loss: {}'.format(i + 1, last_loss))

                if i % 1000 == 999:
                    last_loss = running_loss / (i + 1)
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    
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
                input, labels, confidence, adj = [x.float().to(self.device) for x in data]
                output = self.model(input, adj)
                loss = self.model.compute_loss(output, labels, confidence, self.class_weights)
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


    def save_checkpoint(self, epoch, timestamp):
        directory_path = f'{self.ckpt_dir}/{timestamp}'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        model_path = 'model_{}'.format(epoch)
        torch.save(self.model.state_dict(), f'{directory_path}/{model_path}')
