import torch.nn as nn


def downsample(ch_in, ch_out, batch=True, kernel_size=4, stride=1):
    layers = []
    layers.append(nn.Conv3d(
        ch_in,
        ch_out,
        kernel_size=kernel_size,
        stride=stride,
        bias=not batch
    ))
    if batch:
        layers.append(nn.BatchNorm3d(ch_out))
    layers.append(nn.LeakyReLU(negative_slope=0.02))

    return nn.Sequential(*layers)


def upsample(ch_in, ch_out, dropout=False, kernel_size=4, stride=1):
    layers = []
    layers.append(nn.ConvTranspose3d(
        ch_in,
        ch_out,
        kernel_size=kernel_size,
        stride=stride,
        bias=False
    ))
    layers.append(nn.BatchNorm3d(ch_out))

    if dropout:
        layers.append(nn.Dropout3d(0.5))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)
