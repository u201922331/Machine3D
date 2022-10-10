from enum import Enum
from My3Dimex.vectors import Vec3
from typing import List


class BlockType(Enum):
    none: 0  # Equivalent to 'AIR' blocks
    solid: 1
    vertex: 2  # This is where vertex data is stored, whether there's a single vertex or many


class Block:
    bID: BlockType
    positions: List[Vec3]  # There is the chance that in a single unit there can be more than 1 vertex
    # TODO: Add more properties (if necessary)

    def __init__(self, bID: BlockType, positions: List[Vec3] = None):
        self.update(bID, positions)

    def update(self, newID: BlockType, positions: List[Vec3] = None):
        self.bID = newID
        if newID is BlockType.vertex and positions is None:
            raise ValueError("A block tagged as 'vertex' MUST have at least one position passed through.")
        self.positions = positions
