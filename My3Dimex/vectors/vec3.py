import struct
from typing import List, BinaryIO
import numpy as np


class Vec3:
    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, values: List[float] = None):
        if values is not None:
            self.x, self.y, self.z = values
            return
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f'({self.x}, {self.y}, {self.z})'

    def __len__(self):
        return 3

    def __getitem__(self, item):
        # Currently, it doesn't support slicing
        assert (0 <= item < len(self))
        if item == 0:
            return self.x
        if item == 1:
            return self.y
        if item == 2:
            return self.z

    def __setitem__(self, key, value):
        assert(0 <= key < len(self))
        if key == 0:
            self.x = value
        if key == 1:
            self.y = value
        if key == 2:
            self.z = value

    @property
    def array(self):
        return np.array([self[i] for i in range(len(self))])

    @classmethod
    def read(cls, file: BinaryIO):
        return Vec3(*struct.unpack('fff', file.read(12)))

    def write(self, file: BinaryIO):
        file.write(struct.pack('fff', self.x, self.y, self.z))