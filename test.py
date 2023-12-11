import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
from timeit import default_timer as timer
from utils import *
import random
from datasets.dataset import LoadDataset
from trainer import Trainer
from utils.visualize import Visualization
# datasets = {
#             'sleep-edfx': 0,
#             'HMC': 1,
#             'ISRUC': 2,
#             'SHHS1': 3,
#             'P2018': 3,
#         }


def main():
    parser = argparse.ArgumentParser(description='GSS')
    parser.add_argument('--target_domains', type=str, default='sleep-edfx', help='target_domains')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='dropout')
    parser.add_argument('--datasets_dir', type=str, default='--datasets_dir', help='datasets_dir')
    parser.add_argument('--model_dir', type=str, default='--model_dir', help='model_dir')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0, help='label_smoothing')
    parser.add_argument('--model_path', type=str, default='--model_path', help='model_path')

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)

    torch.cuda.set_device(params.cuda)

    visualization = Visualization(params)
    # visualization.visualize()
    # visualization.visualize_correlation()
    visualization.visualize_cor_seaborn()



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()
