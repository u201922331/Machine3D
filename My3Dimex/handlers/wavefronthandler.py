import sys
from dataclasses import dataclass
from typing import List
import numpy as np
from .stlhandler import STL
from ..vectors import Vec2, Vec3
from ..triangle import Triangle


@dataclass
class Wavefront:
    positions: List[Vec3]
    texCoords: List[Vec2]
    normals: List[Vec3]
    faceIndices: List[List[List[int]]]

    def __repr__(self):
        return f'Positions({len(self.positions)}): {self.positions}\n' \
               f'Texture Coordinates({len(self.texCoords)}): {self.texCoords}\n' \
               f'Normals({len(self.normals)}): {self.normals}\n' \
               f'Face Indices({len(self.faceIndices)}): {self.faceIndices}'

    @classmethod
    def read(cls, filename: str):
        _positions = []
        _texCoords = []
        _normals = []
        _faceIndices = []

        with open(filename) as file:
            for line in file.readlines():
                line = line.split()
                if len(line) < 1:
                    continue
                tag, contents = line[0], line[1:]

                if tag == 'v':
                    _positions.append(Vec3(values=list(map(float, contents))))
                if tag == 'vt':
                    _texCoords.append(Vec2(values=list(map(float, contents))))
                if tag == 'vn':
                    _normals.append(Vec3(values=list(map(float, contents))))
                if tag == 'f':
                    vertices = [[int(idx)-1 for idx in content.split('/')] for content in contents]
                    if len(vertices) == 3:
                        _faceIndices.append(vertices)
                    if len(vertices) == 4:
                        _faceIndices.append(vertices[:3])
                        _faceIndices.append([vertices[0], vertices[2], vertices[3]])
            return Wavefront(_positions, _texCoords, _normals, _faceIndices)

    def toSTL(self):
        output = STL(f'Made in Python {sys.version}', [])
        for face in self.faceIndices:
            _p, _tC, _n = list(zip(*face))
            p0, p1, p2 = [self.positions[p] for p in _p]
            n = Vec3(values=np.sum([self.normals[n].array for n in _n], axis=0)/3)
            output.triangles.append(Triangle(n, [p0, p1, p2], 0))
        return output
