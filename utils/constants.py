import torch.nn as nn

PATH = './'
PATH_MODELS = PATH + '/models'
PATH_MESHES = PATH + '/meshes'
PATH_RESULTS = PATH + '/results'
PATH_CHECKPOINTS_GEN = PATH + '/checkpoints/gen'
PATH_CHECKPOINTS_DIS = PATH + '/checkpoints/dis'

learning_rate = 0.00001
epochs = 10

LAMBDA = 100

display_step = 48
batch_size = 50

criterion = nn.BCEWithLogitsLoss()
