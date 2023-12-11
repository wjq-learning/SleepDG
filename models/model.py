import torch
import torch.nn as nn
from models.encoder import Encoder
from torch.autograd import Function
from models.ae import AE

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.vae = AE(params)
        self.classifier = nn.Linear(512, self.params.num_of_classes)

    def forward(self, x):
        recon, mu = self.vae(x)
        return self.classifier(mu), recon, mu

    def inference(self, x):
        mu = self.vae.encoder(x)
        return self.classifier(mu)


