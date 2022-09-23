import struct
from dataclasses import dataclass
from typing import List, BinaryIO
import numpy as np
from . import vectors as vct


@dataclass
class Triangle:
    normal: vct.Vec3
    vertices: List[vct.Vec3]
    attrib: int

    def __repr__(self):
        return f"{dict(zip(('Normal', 'Vertices', 'Attrib'), (self.normal, self.vertices, self.attrib)))}"

    @classmethod
    def read(cls, file: BinaryIO):
        normal = vct.Vec3.read(file)
        vertices = [vct.Vec3.read(file) for _ in range(3)]
        attrib, = struct.unpack('H', file.read(2))
        return Triangle(normal, vertices, attrib)

    def write(self, file: BinaryIO):
        self.normal.write(file)
        for vertex in self.vertices:
            vertex.write(file)
        file.write(struct.pack('H', self.attrib))

    @classmethod
    def generate(cls, p0: vct.Vec3, p1: vct.Vec3, p2: vct.Vec3, a: int):
        n = np.cross(p2.array-p1.array, p2.array-p0.array)
        n = vct.Vec3(values=n/np.linalg.norm(n))
        return Triangle(n, [p0, p1, p2], a)
