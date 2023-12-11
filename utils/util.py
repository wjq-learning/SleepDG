import torch
import numpy as np

def to_tensor(array):
    return torch.from_numpy(np.array(array)).float()