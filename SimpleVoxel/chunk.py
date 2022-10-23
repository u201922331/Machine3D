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
    # __offset: Vec3

    def __init__(self, w: int, h: int, d: int):
        self.__w = w
        self.__h = h
        self.__d = d
        self.reset()

    @property
    def shape(self):
        return self.__h, self.__d, self.__w

    def __getitem__(self, item: List[int]):
        x, y, z = item
        return self.blocks[y][z][x]

    def __setitem__(self, key: List[int], value: Block):
        x, y, z = key
        self.blocks[y][z][x] = value

    def update(self, vertices):
        # Convertir y almancenar los vertices en bloques
        pass

    def reset(self):
        self.blocks = [[[Block(BlockType.none)
                         for _ in range(self.__w)]
                        for _ in range(self.__d)]
                       for _ in range(self.__h)]

    def plot(self, col='#2A3445', yUp: bool = True):
        colorSolid = np.empty(self.shape, dtype=bool)
        for h in range(self.__h):
            for d in range(self.__d):
                for w in range(self.__w):
                    colorSolid[h][d][w] = self.blocks[h][d][w].bID is BlockType.solid

        voxels = colorSolid
        colors = np.empty(voxels.shape, dtype=object)
        colors[colorSolid] = col

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolors='k')
        ax.axis('equal')
        plt.show()
