import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import Model
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_accuracy(self, model):
        model.eval()

        truths = []
        preds = []
        for x, y, z in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model.inference(x)
            y = y.view(-1)
            pred = pred.view(-1, 5)
            pred_y = torch.max(pred, dim=1)[1]

            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        # print(truths.shape)
        # print(preds.shape)
        acc = accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average="macro")
        cm = confusion_matrix(truths, preds)
        wake_f1 = f1_score(truths==0, preds==0)
        n1_f1 = f1_score(truths==1, preds==1)
        n2_f1 = f1_score(truths==2, preds==2)
        n3_f1 = f1_score(truths==3, preds==3)
        rem_f1 = f1_score(truths==4, preds==4)


        return acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1