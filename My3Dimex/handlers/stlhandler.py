import struct
from dataclasses import dataclass
from typing import List
from ..triangle import Triangle
from copy import deepcopy
import numpy as np
import pandas as pd
import sys
from ..vectors import Vec3


@dataclass
class STL:
    header: str
    triangles: List[Triangle]

    def __repr__(self):
        return f"Header({len(self.header)}): {self.header}\n" \
               f"Triangle count: {len(self.triangles)}\n" \
               f"Triangles: {self.triangles}"

    @classmethod
    def read(cls, filename: str):
        _header = ""
        _triangles = []
        with open(filename, 'rb') as file:
            for elem in file.read(80).decode():
                if elem != chr(0):
                    _header += elem
            triCount, = struct.unpack('I', file.read(4))
            _triangles = [Triangle.read(file) for _ in range(triCount)]
            file.close()
        return STL(_header, _triangles)

    def write(self, filename: str):
        with open(filename, 'wb') as file:
            _header = deepcopy(self.header)
            _header += chr(0) * np.max(80-len(self.header), 0)
            headerBytes = _header[:80].encode()
            file.write(headerBytes)
            file.write(struct.pack('I', len(self.triangles)))
            for triangle in self.triangles:
                triangle.write(file)
            file.close()

    def toDataFrame(self):
        Normals: List[Vec3] = [t.normal for t in self.triangles]
        P0: List[Vec3] = [t.vertices[0] for t in self.triangles]
        P1: List[Vec3] = [t.vertices[1] for t in self.triangles]
        P2: List[Vec3] = [t.vertices[2] for t in self.triangles]
        Attributes = [t.attrib for t in self.triangles]
        return pd.DataFrame({'Normal': Normals,
                             'P0': P0,
                             'P1': P1,
                             'P2': P2,
                             'Attribute': Attributes})

    @classmethod
    def fromDataFrame(cls, df: pd.DataFrame):
        output = STL(f"Made in Python {sys.version}", [])
        for _, t in df.iterrows():
            output.triangles.append(Triangle(t['Normal'],
                                             [t['P0'],
                                              t['P1'],
                                              t['P2']],
                                             t['Attribute']))
        return output
