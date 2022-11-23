import torch
import torch.nn as nn
from sampling import downsample, upsample


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ff = 10

        self.down_stack = [
            downsample(1, ff, batch=False),
            downsample(ff, 2 * ff),
            downsample(2 * ff, 4 * ff, stride=2),
            downsample(4 * ff, 4 * ff, stride=2),
            downsample(4 * ff, 4 * ff, stride=2, batch=False)
        ]

        self.up_stack = [
            upsample(4 * ff, 4 * ff, dropout=False, kernel_size=5),
            upsample(8 * ff, 4 * ff, dropout=False, stride=2),
            upsample(8 * ff, 2 * ff, stride=2),
            upsample(4 * ff, ff)
        ]

        self.result = nn.ConvTranspose3d(2 * ff, 1, kernel_size=4)
        self.act = nn.Tanh()

    def forward(self, x):
        s = []

        for elem in self.down_stack:
            x = elem(x)
            s.append(x)

        for i in range(len(self.up_stack)):
            x = self.up_stack[i](x)
            x = torch.cat((x, s[-(i + 2)]), dim=1)

        return self.act(self.result(x))