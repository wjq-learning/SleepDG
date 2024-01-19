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
from losses.double_alignment import CORAL
from losses.ae_loss import AELoss
from timeit import default_timer as timer
import os
import copy

class Trainer(object):
    def __init__(self, params):
        self.params = params
        self.data_loader, subject_id = LoadDataset(params).get_data_loader()

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.best_model_states = None

        self.model = Model(params).cuda()
        self.ce_loss = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        self.coral_loss = CORAL().cuda()
        self.ae_loss = AELoss().cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.lr/10)

        self.data_length = len(self.data_loader['train'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.params.epochs * self.data_length
        )

        print(self.model)

    def train(self):
        acc_best = 0
        f1_best = 0
        i = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y, z in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                z = z.cuda()
                # print(z)
                pred, recon, mu = self.model(x)
                loss1 = self.ce_loss(pred.transpose(1, 2), y)
                loss2 = self.coral_loss(mu, z)
                loss3 = self.ae_loss(x, recon)
                # print(loss1, loss2, loss3)
                loss = loss1 + loss2 + loss3
                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.scheduler.step()
            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1 = self.val_eval.get_accuracy(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                print(
                    "wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
                        wake_f1,
                        n1_f1,
                        n2_f1,
                        n3_f1,
                        rem_f1,
                    )
                )
                if acc > acc_best:
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    f1_best = f1
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
                    print("Epoch {}: ACC increasing!! New acc: {:.5f}, f1: {:.5f}".format(best_f1_epoch, acc_best, f1_best))
        print("{} epoch get the best acc {:.5f} and f1 {:.5f}".format(best_f1_epoch, acc_best, f1_best))
        test_acc, test_f1 = self.test()
        return test_acc, test_f1

    def test(self):
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
            test_n3_f1, test_rem_f1 = self.test_eval.get_accuracy(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, f1: {:.5f}".format(
                    test_acc,
                    test_f1,
                )
            )
            print(test_cm)
            print(
                "wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
                    test_wake_f1,
                    test_n1_f1,
                    test_n2_f1,
                    test_n3_f1,
                    test_rem_f1,
                )
            )
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/tacc_{:.5f}_tf1_{:.5f}.pth".format(
                test_acc,
                test_f1,
            )
            torch.save(self.best_model_states, model_path)
            print("the model is save in " + model_path)
        return test_acc, test_f1
