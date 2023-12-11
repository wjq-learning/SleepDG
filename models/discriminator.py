import torch
import torch.nn as nn
from models.transformer import TransformerEncoder


class Discriminator(nn.Module):
    def __init__(self, params, subject_id):
        super(Discriminator, self).__init__()
        self.params = params
        self.avgpooling = nn.AdaptiveAvgPool1d(1)

        self.subject_mapping = TransformerEncoder(
                seq_length=20,
                num_layers=1,
                num_heads=8,
                hidden_dim=512,
                mlp_dim=512,
                dropout=self.params.dropout,
                attention_dropout=self.params.dropout,
            )
        self.subject_classifier = nn.Linear(512, subject_id)
        self.grl = GRL()

    def forward(self, x):
        x = self.grl(x)
        x = self.subject_mapping(x)
        x = self.avgpooling(x.transpose(1, 2)).squeeze()
        return self.subject_classifier(x)


class GRL(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)


class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None
