import torch
import torch.nn as nn
import pickle
import numpy as np
from tqdm import tqdm
import glob
import h5py
import json
import matplotlib.pyplot as plt
import math
from dataset import AutoProofDataset, build_dataloader
from model import create_model

def main():
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    config['loader']['num_workers'] = 32
    config['loader']['batch_size'] = 32
    data = AutoProofDataset(config)
    train_loader, val_loader, train_size, val_size = build_dataloader(data, config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = config['loader']['batch_size']

    model = create_model(config)
    model_num = 'data/ckpt/20241104_132019/model_29'
    model.load_state_dict(torch.load(f'../../{model_num}'))
    model.to(device)
    model.eval()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    no_merge_prob = []
    merge_prob = []
    merge_prob_other = []

    with tqdm(total=val_size / batch_size, desc="val") as pbar:
        for i, data in enumerate(val_loader):
            input, labels, confidence, adj = [x.float().to(device) for x in data]
            output = model(input, adj)
            sigmoid = nn.Sigmoid()
            output = sigmoid(output)

            mask = labels != -1
            mask = mask.squeeze(-1)
            output = output[mask]
            labels = labels[mask].long().squeeze(-1)

            # if len(torch.where(labels == 0, dim=-1)[0]):
            #     print("STOPPP, GOT ALL 1 LABEL")
            mask_0 = labels == 0
            mask_1 = labels == 1

            curr_merge_prob = output[mask_0,0].tolist()
            curr_no_merge_prob = output[mask_1,1].tolist()
            curr_merge_prob_other = output[mask_1, 0].tolist()
            # for j in range(len(labels)):
            #     if labels[j] == 0:
            #         merge_prob.append(output[j][labels[j]].item())
            #     else:
            #         no_merge_prob.append(output[j][labels[j]].item())
            # print("output", output)
            # print("labels", labels)
            merge_prob.extend(curr_merge_prob)
            no_merge_prob.extend(curr_no_merge_prob)
            merge_prob_other.extend(curr_merge_prob_other)
            

            output = torch.argmax(output, dim=-1)
            curr_tp, curr_tn, curr_fp, curr_fn = model.accuracy(output, labels)
            tp += curr_tp
            tn += curr_tn
            fp += curr_fp
            fn += curr_fn

            
            # print("output", output.squeeze(-1).long())
            # print("labels", labels.squeeze(-1).long())

            pbar.update()

        # no_merge_prob = np.array(no_merge_prob).flatten()
        # merge_prob = np_array(merge_prob).flatten()
        no_merge_prob = np.array(no_merge_prob)
        merge_prob = np.array(merge_prob)
        merge_prob_other = np.array(merge_prob_other)
        print("no merge prob len", no_merge_prob.shape)
        print("merge prob len", merge_prob.shape)

        # no merge prob dist
        plt.hist(no_merge_prob, bins=10, edgecolor='black')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title('No Merge Error Probabilty')
        plt.savefig(f'../../data/figures/no_merge_prob_20241104_132019_model_29')
        plt.close()

        plt.hist(merge_prob, bins=10, edgecolor='black')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title('Merge Error Probabilty when Label is Merge Error')
        plt.savefig(f'../../data/figures/merge_prob_when_merge_20241104_132019_model_29')
        plt.close()
        
        plt.hist(merge_prob_other, bins=10, edgecolor='black')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title('Merge Error Probabilty When Label is No Merge Error')
        plt.savefig(f'../../data/figures/merge_prob_when_no_merge_20241104_132019_model_29')
        plt.close()

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
        
        print("f1", f1)
        print("recall", recall)
        print("g_mean", g_mean)

def one_class():
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    config['loader']['num_workers'] = 1
    config['loader']['batch_size'] = 1
    data = AutoProofDataset(config)
    train_loader, val_loader, train_size, val_size = build_dataloader(data, config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = config['loader']['batch_size']

    model = create_model(config)
    model.load_state_dict(torch.load('../../data/ckpt/20241103_023121/model_66'))
    model.to(device)
    model.eval()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    with tqdm(total=val_size / batch_size, desc="val") as pbar:
        for i, data in enumerate(val_loader):
            input, labels, confidence, adj = [x.float().to(device) for x in data]
            output = model(input, adj)
            # output = torch.argmax(output, dim=-1)
            # condfidence = 0.5
            # output[output >= confidence] = 1
            # output[output < confidence] = 0


            curr_tp, curr_tn, curr_fp, curr_fn = model.accuracy(output, labels)
            tp += curr_tp
            tn += curr_tn
            fp += curr_fp
            fn += curr_fn

            mask = labels != -1
            mask = mask.squeeze(-1)
            output = output[mask]
            labels = labels[mask]
            print("output", output.squeeze(-1).long())
            print("labels", labels.squeeze(-1).long())

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
        
        print("f1", f1)
        print("recall", recall)
        print("g_mean", g_mean)

def plot_loss():
    loss_history = {
        0.0001: 0.5570,
        0.0005: 0.4990,
        0.001: 0.3760,
        0.005: 0.3760,
        0.1: 0.2370,
        1: 0.2340,
        2: 0.2305,
        3: 0.2285,
        4: 0.2267,
        5: 0.2242,
        6: 0.1968,
        7: 0.1524,
        8: 0.1427,
        9: 0.1380,
        10: 0.1352,
        11: 0.1333,
        12: 0.1319,
        13: 0.1308,
        14: 0.1298,
        15: 0.1290,
        16: 0.1283,
        17: 0.1276,
        18: 0.1267,
        30: 0.1222
    }
    epochs = list(loss_history.keys())
    losses = list(loss_history.values())

    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.grid(True)
    plt.savefig(f'../../data/figures/loss_20241104_132019_model_29')
    plt.close()

if __name__ == "__main__":
    main()
    # one_class()
    plot_loss()