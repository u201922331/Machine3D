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
        # Random blocks (for testing purposes)
        """
        self.blocks = [[[Block(np.random.choice(list(BlockType), p=[0.5, 0.45, 0.05]),
                               [VertexInfo(Vec3(), 0)])
                         for _ in range(w)]
                        for _ in range(d)]
                       for _ in range(h)]
        """
        # We initialize the blocks list as a bunch of empty spaces
        self.blocks = [[[Block(BlockType.none) for _ in range(w)] for _ in range(d)] for _ in range(h)]

    @property
    def shape(self) -> tuple:
        return self.__w, self.__h, self.__d

    def __getitem__(self, item: List[int]):
        x, y, z = item
        return self.blocks[y][z][x]

    def __setitem__(self, key: List[int], value: Block):
        x, y, z = key
        self.blocks[y][z][x] = value

    def plot(self, c1='#4579CC', c2='#2A3445'):
        x, y, z = np.indices(self.shape)

        colVtx = np.empty(self.shape, dtype=bool)
        colMsh = np.empty(self.shape, dtype=bool)
        for i in range(self.__h):
            for j in range(self.__d):
                for k in range(self.__w):
                    colVtx[i][j][k] = self.blocks[i][j][k].bID is BlockType.vertex
                    colMsh[i][j][k] = self.blocks[i][j][k].bID is BlockType.solid

        voxels = colVtx | colMsh
        colors = np.empty(voxels.shape, dtype=object)
        colors[colVtx] = c1
        colors[colMsh] = c2

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolors='k')
        plt.show()

        pass
