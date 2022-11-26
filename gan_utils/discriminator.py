import torch
import torch.nn as nn
from .sampling import down


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d1 = down(2, 16, batch=False)
        self.d2 = down(16, 32)
        self.d3 = down(32, 64, stride=2)
        self.d4 = down(64, 64, stride=2)

        self.res = nn.ConvTranspose3d(64, 1, 4)
        self.activator = nn.SiLU()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)

        for elem in (self.d1, self.d2, self.d3, self.d4):
            x = elem(x)

        return self.activator(self.res(x))
