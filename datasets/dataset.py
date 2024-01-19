import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random


class CustomDataset(Dataset):
    def __init__(
            self,
            seqs_labels_path_pair
    ):
        super(CustomDataset, self).__init__()
        self.seqs_labels_path_pair = seqs_labels_path_pair

    def __len__(self):
        return len((self.seqs_labels_path_pair))

    def __getitem__(self, idx):
        seq_path = self.seqs_labels_path_pair[idx][0]
        label_path = self.seqs_labels_path_pair[idx][1]
        subject_id = self.seqs_labels_path_pair[idx][2]
        seq_eeg = np.load(seq_path)[:, :1, :]
        seq_eog = np.load(seq_path)[:, 1:2, :]
        seq = np.concatenate((seq_eeg, seq_eog), axis=1)
        label = np.load(label_path)
        return seq, label, subject_id

    def collate(self, batch):
        x_seq = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        z_label = np.array([x[2] for x in batch])
        return to_tensor(x_seq), to_tensor(y_label).long(), to_tensor(z_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets = {
            'sleep-edfx': 0,
            'HMC': 1,
            'ISRUC': 2,
            'SHHS1': 3,
            'P2018': 4,
        }
        self.targets_dirs = [f'{self.params.datasets_dir}/{item}' for item in self.datasets.keys() if item in self.params.target_domains]
        self.source_dirs = [f'{self.params.datasets_dir}/{item}' for item in self.datasets.keys() if item not in self.params.target_domains]
        print(self.targets_dirs)
        print(self.source_dirs)

    def get_data_loader(self):
        source_domains, subject_id = self.load_path(self.source_dirs, 0)
        target_domains, _ = self.load_path(self.targets_dirs, subject_id)
        # print(len(target_domains), len(source_domains))
        train_pairs, val_pairs = self.split_dataset(source_domains)
        print(len(train_pairs), len(val_pairs), len(target_domains))
        train_set = CustomDataset(train_pairs)
        val_set = CustomDataset(val_pairs)
        test_set = CustomDataset(target_domains)
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=self.params.num_workers,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=True,
                num_workers=self.params.num_workers,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=True,
                num_workers=self.params.num_workers,
            ),
        }
        return data_loader, subject_id

    def load_path(self, domains_dirs, subject_id):
        domains = []
        for dataset in domains_dirs:
            seq_dirs = os.listdir(f'{dataset}/seq')
            labels_dirs = os.listdir(f'{dataset}/labels')
            for seq_dir, labels_dir in zip(seq_dirs, labels_dirs):
                seq_names = os.listdir(os.path.join(dataset, 'seq', seq_dir))
                labels_names = os.listdir(os.path.join(dataset, 'labels', labels_dir))
                for seq_name, labels_name in zip(seq_names, labels_names):
                    domains.append((os.path.join(dataset, 'seq', seq_dir, seq_name), os.path.join(dataset, 'labels', labels_dir, labels_name), subject_id))
            subject_id += 1
        return domains, subject_id

    def split_dataset(self, source_domains):
        random.shuffle(source_domains)
        split_num = int(len(source_domains) * 0.8)
        train_pairs = source_domains[:split_num]
        val_pairs = source_domains[split_num:]
        return train_pairs, val_pairs



if __name__ == '__main__':
    import argparse

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    parser = argparse.ArgumentParser(description='SleepDG')
    parser.add_argument('--target_domains', type=list, default=['SHHS1'], help='target_domains')
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=7, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='dropout')
    parser.add_argument('--datasets_dir', type=str, default='--datasets_dir', help='datasets_dir')
    parser.add_argument('--model_dir', type=str, default='--model_dir', help='model_dir')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')

    params = parser.parse_args()
    setup_seed(params.seed)

    loadDataset = LoadDataset(params)
    loadDataset.get_data_loader()
    data_loader, _ = loadDataset.get_data_loader()
    print(_)
    print(data_loader)
    print(len(data_loader['train']))
    #
    # for x, y in data_loader['train']:
    #     print(x.shape)
    #     print(y.shape)