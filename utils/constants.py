import torch.nn as nn

ROOT = './'
MODELS = ROOT + 'models/'
MESHES = ROOT + 'meshes/'
DATASETS = ROOT + 'datasets/'

lr = 0.0002

batch_size = 50

criterion = nn.BCEWithLogitsLoss()

tseed_ = 1234
