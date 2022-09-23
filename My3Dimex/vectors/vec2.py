import struct
from typing import List, BinaryIO
import numpy as np


class Vec2:
    x: float
    y: float

    def __init__(self, x: float = 0.0, y: float = 0.0, values: List[float] = None):
        if values is not None:
            self.x, self.y = values
            return
        self.x = x
        self.y = y

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __len__(self):
        return 2

    def __getitem__(self, item):
        # Currently, it doesn't support slicing
        assert (0 <= item < len(self))
        if item == 0:
            return self.x
        if item == 1:
            return self.y

    def __setitem__(self, key, value):
        assert(0 <= key < len(self))
        if key == 0:
            self.x = value
        if key == 1:
            self.y = value

    @property
    def array(self):
        return np.array([self[i] for i in range(len(self))])

    @classmethod
    def read(cls, file: BinaryIO):
        return Vec2(*struct.unpack('ff', file.read(8)))

    def write(self, file: BinaryIO):
        file.write(struct.pack('ff', self.x, self.y))
