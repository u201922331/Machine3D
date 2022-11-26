import torch
import torch.nn as nn
from sampling import down, up


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.d1 = down(1, 10, batch=False)
        self.d2 = down(10, 20)
        self.d3 = down(20, 40, stride=2)
        self.d4 = down(40, 40, stride=2)
        self.d5 = down(40, 40, stride=2, batch=False)

        self.u1 = up(40, 40, kernel_size=5)
        self.u2 = up(80, 40, stride=2)
        self.u3 = up(80, 20, stride=2)
        self.u4 = up(40, 10)

        self.res = nn.ConvTranspose3d(20, 1, 4)
        self.activator = nn.Tanh()

    def forward(self, x):
        s = []
        for elem in (self.d1, self.d2, self.d3, self.d4, self.d5):
            x = elem(x)
            s.append(x)

        s = s[:-1]

        x = self.u1(x)
        x = torch.cat((x, s[-1]), dim=1)
        x = self.u2(x)
        x = torch.cat((x, s[-2]), dim=1)
        x = self.u3(x)
        x = torch.cat((x, s[-3]), dim=1)
        x = self.u4(x)
        x = torch.cat((x, s[-4]), dim=1)

        return self.activator(self.res(x))
