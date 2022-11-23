import torch
import torch.nn as nn
from sampling import downsample


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.down_stack = [
            downsample(2, 16, batch=False),
            downsample(16, 32),
            downsample(32, 64, stride=2),
            downsample(64, 64, stride=2),
            downsample(64, 64, stride=2)
        ]

        self.result = nn.ConvTranspose3d(64, 1, kernel_size=4)
        self.act = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)

        for elem in self.down_stack:
            x = elem(x.float())

        return self.act(self.result(x))
