import matplotlib.pyplot as plt
import numpy as np
from typing import List
from .block import VertexInfo, BlockType, Block
from My3Dimex.vectors import Vec3


class Chunk:
    blocks: List[List[List[Block]]]
    __w: int
    __h: int
    __d: int
    __offset: Vec3

    def __init__(self, w: int, h: int, d: int):
        self.__w = w
        self.__h = h
        self.__d = d
        self.blocks = [[[Block(np.random.choice(list(BlockType)),
                               [VertexInfo(Vec3(), 0)])
                         for _ in range(w)]
                        for _ in range(d)]
                       for _ in range(h)]

    @property
    def shape(self) -> tuple:
        return self.__w, self.__h, self.__d

    def __getitem__(self, item: List[int]):
        x, y, z = item
        return self.blocks[y][z][x]

    def __setitem__(self, key: List[int], value):
        x, y, z = key
        self.blocks[y][z][x] = value

    def plot(self, c1='#405F91', c2='#2A3445'):
        x, y, z = np.indices(self.shape)
        print(np.array(self.blocks, dtype=Block)[0])
        colVtx = np.array(self.blocks)[y][z][x].bID is BlockType.vertex  # Fix this
        colMsh = np.array(self.blocks)[y][z][x].bID is BlockType.solid  # Fix this
        voxels = colVtx | colMsh
        colors = np.empty(voxels.shape)
        colors[colVtx] = c1
        colors[colMsh] = c2

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolors='k')
        plt.show()

        pass
