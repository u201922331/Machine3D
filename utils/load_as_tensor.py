import torch
import numpy as np
from .binvox_ops import binvox_to_3d_array


def load_binvox_as_tensor(path, device):
    return torch.from_numpy(np.array([binvox_to_3d_array(path)])*2-1).float().to(device)
