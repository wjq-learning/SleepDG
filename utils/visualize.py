import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import Model
from tqdm import tqdm
import numpy as np
import torch
from datasets.dataset import LoadDataset
from evaluator import Evaluator
from torch.nn import CrossEntropyLoss
from timeit import default_timer as timer
import os
from sklearn import manifold
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FastICA
import matplotlib.pyplot as plt
import matplotlib
import umap
import seaborn as sns
import pandas as pd
# sns.set(style="darkgrid")
label2color = {
    0: 'yellow',
    1: 'red',
    2: 'blue',
    3: 'pink',
    4: 'green',
}
label2class = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM',
}
font1 = {'family': 'Times New Roman'}
matplotlib.rc("font", **font1)
class Visualization(object):
    def __init__(self, params):
        self.params = params
        self.data_loader, subject_id = LoadDataset(params).get_data_loader()

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = Model(params).cuda()
        self.model.load_state_dict(torch.load(self.params.model_path, map_location='cpu'))
        print(self.model)

    def visualize(self):
        # ts = manifold.TSNE(n_components=2, random_state=42)
        ts = umap.UMAP(random_state=42)
        self.model.eval()
        feats_list = []
        labels_list = []
        i = 0
        for x, y, z in tqdm(self.data_loader['train']):
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            pred, recon, mu = self.model(x)
            # print(seq_features.shape)
            feats = mu.view(-1, 512)
            feats_list.append(feats.detach().cpu().numpy())
            labels = y.view(-1)
            labels_list.append(labels.detach().cpu().numpy())
            i += 1
            if i > 10:
                break
        feats_all = numpy.concatenate(feats_list, axis=0)
        labels_all = numpy.concatenate(labels_list, axis=0)


        feats_ts = ts.fit_transform(feats_all)
        print(feats_ts.shape)

        self.draw(feats_ts, labels_all)


    def draw(self, feats_ts, labels_all):
        xs = [[], [], [], [], []]
        ys = [[], [], [], [], []]
        for i, label in enumerate(labels_all):
            xs[label].append(feats_ts[i, 0])
            ys[label].append(feats_ts[i, 1])

        Wake = plt.scatter(x=xs[0], y=ys[0], c='yellow', alpha=1, marker='.', label='Wake')
        N1 = plt.scatter(x=xs[1], y=ys[1], c='red', alpha=1, marker='.', label='N1')
        N2 = plt.scatter(x=xs[2], y=ys[2], c='blue', alpha=1, marker='.', label='N2')
        N3 = plt.scatter(x=xs[3], y=ys[3], c='pink', alpha=1, marker='.', label='N3')
        REM = plt.scatter(x=xs[4], y=ys[4], c='green', alpha=1, marker='.', label='REM')
        plt.xticks([], fontsize=20)
        plt.yticks([], fontsize=20)
        plt.legend(fontsize=12)
        plt.show()
        # plt.savefig(f"visualize2.pdf", bbox_inches='tight', pad_inches=0.01)

    def visualize_correlation(self):
        ts = manifold.Isomap(n_components=1)
        # ts = PCA(n_components=1, random_state=42)
        self.model.eval()
        seqs_list = [[], [], [], [], []]
        # domains_list = []
        i = 0
        for x, y, z in tqdm(self.data_loader['train']):
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            pred, recon, mu = self.model(x)
            bz = z.shape[0]
            for i in range(bz):
                seqs_list[z[i]].append(mu[i].detach().cpu().numpy())
        for x, y, z in tqdm(self.data_loader['test']):
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            pred, recon, mu = self.model(x)
            bz = z.shape[0]
            for i in range(bz):
                seqs_list[z[i]].append(mu[i].detach().cpu().numpy())
        means_list = []
        for domain_id in range(5):
            means_list.append(np.mean(np.array(seqs_list[domain_id]), axis=0))

        means_array = np.concatenate(means_list, axis=0)
        print(means_array.shape)
        means_ts = ts.fit_transform(means_array)
        print(means_ts.shape)
        means_ts = means_ts.reshape(5, 20)
        self.draw_cor(means_ts)

    def draw_cor(self, means_ts):
        plt.plot(means_ts[4], c='pink', alpha=1, marker='.', label='SleepEDFx')
        plt.plot(means_ts[0], c='green', alpha=1, marker='.', label='HMC')
        plt.plot(means_ts[1], c='yellow', alpha=1, marker='.', label='ISRUC')
        plt.plot(means_ts[2], c='red', alpha=1, marker='.', label='SHHS')
        plt.plot(means_ts[3], c='blue', alpha=1, marker='.', label='P2018')
        plt.legend(fontsize=12)
        plt.ylim(-7, 7)
        plt.show()

    def visualize_cor_seaborn(self):
        # ts = manifold.Isomap(n_components=1)
        ts = PCA(n_components=1, random_state=42)
        self.model.eval()
        seqs_list = []
        domains_list = []
        n = 0
        for x, y, z in tqdm(self.data_loader['train']):
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            pred, recon, mu = self.model(x)
            bz = z.shape[0]
            seqs_list.append(mu.detach().cpu().numpy())
            domains_list.append(z.detach().cpu().numpy())
            n += 1
            if n > 100:
                break
        n = 0
        for x, y, z in tqdm(self.data_loader['test']):
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            pred, recon, mu = self.model(x)
            bz = z.shape[0]
            seqs_list.append(mu.detach().cpu().numpy())
            domains_list.append(z.detach().cpu().numpy())
            n += 1
            if n > 100:
                break
        seqs_array = np.concatenate(seqs_list, axis=0)
        domains_array = np.concatenate(domains_list, axis=0)
        print(seqs_array.shape, domains_array.shape)
        seqs_array = seqs_array.reshape(-1, 512)
        seqs_ts = ts.fit_transform(seqs_array)
        seqs_ts = seqs_ts.reshape(-1, 20)
        print(seqs_ts.shape)

        # print(means_ts.shape)
        # means_ts = means_ts.reshape(5, 20)
        self.draw_cor_seaborn(seqs_ts, domains_array)

    def draw_cor_seaborn(self, seqs_ts, domains_array):
        id2domain = ['HMC', 'ISRUC', 'SHHS', 'P2018','SleepEDFx']
        data_list = []
        for i in range(seqs_ts.shape[0]):
            for j in range(20):
                data_list.append({
                    'feature': seqs_ts[i][j],
                    'time': j,
                    'Domains': id2domain[domains_array[i]]
                })
        df = pd.DataFrame(data_list)
        sns.lineplot(x="time", y="feature",hue="Domains", style="Domains", data=df)
        # plt.xticks([], fontsize=20)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([], fontsize=20)
        plt.yticks([], fontsize=20)
        plt.ylim(-1.6, 1)
        plt.legend(fontsize=12)
        # plt.show()
        plt.savefig(f"visualize4.pdf", bbox_inches='tight', pad_inches=0.01)
