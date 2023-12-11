import torch

a = torch.randn((4, 3, 5))
b = a.mean(1)
print(b.shape)