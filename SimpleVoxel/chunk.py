from .block import *
from typing import List


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
        self.blocks = [[[Block(BlockType.none) for _ in range(w)] for _ in range(d)] for _ in range(h)]

    def __len__(self):
        return self.__w, self.__h, self.__d

    def __call__(self, x: int, y: int, z: int):
        return self.getBlock(x, y, z)

    def getBlock(self, x: int, y: int, z: int):
        return self.blocks[y][z][x]
