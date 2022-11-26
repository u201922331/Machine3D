import torch.nn as nn


def down(i, o, batch=True, kernel_size=4, stride=1):
    layers = []
    layers.append(nn.Conv3d(
        i, o,
        kernel_size=kernel_size,
        stride=stride,
        bias=not batch
    ))
    if batch:
        layers.append(nn.BatchNorm3d(o))
    layers.append(nn.LeakyReLU(0.02))

    return nn.Sequential(*layers)


def up(i, o, dropout=False, kernel_size=4, stride=1):
    layers = []
    layers.append(nn.ConvTranspose3d(
        i, o,
        kernel_size=kernel_size,
        stride=stride,
        bias=False
    ))
    layers.append(nn.BatchNorm3d(o))

    if dropout:
        layers.append(nn.Dropout())
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)
