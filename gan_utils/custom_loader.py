import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.constants import *


def load(dataset_path):
    models = np.load(dataset_path, allow_pickle=True).item()
    models_tr_x = [torch.from_numpy(np.array([model]) * 2 - 1).float() for model in models['train']['x']]
    models_tr_y = [torch.from_numpy(np.array([model]) * 2 - 1).float() for model in models['train']['y']]
    models_ts_x = [torch.from_numpy(np.array([model]) * 2 - 1).float() for model in models['test']['x']]
    models_ts_y = [torch.from_numpy(np.array([model]) * 2 - 1).float() for model in models['test']['y']]

    dl_tr = DataLoader(list(zip(models_tr_x, models_tr_y)), batch_size=batch_size, shuffle=True)
    dl_ts = DataLoader(list(zip(models_ts_x, models_ts_y)), batch_size=batch_size, shuffle=True)

    return dl_tr, dl_ts, models_tr_x, models_tr_y, models_ts_x, models_ts_y
