from enum import Enum
from typing import List
from dataclasses import dataclass
from My3Dimex.vectors import Vec3


@dataclass
class VertexInfo:
    position: Vec3  # Position of the vertex
    gIndex: int  # Global index of the vertex (used later when reconstructing the model from voxel data)


class BlockType(Enum):
    none = 0  # Equivalent to 'AIR' blocks
    solid = 1
    vertex = 2  # This is where vertex data is stored, whether there's a single vertex or many


class Block:
    bID: BlockType
    positions: List[VertexInfo]  # There is the chance that in a single unit there can be more than 1 vertex
    # TODO: Add more properties (if necessary)

    def __init__(self, bID: BlockType, positions: List[VertexInfo] = None):
        self.update(bID, positions)

    def update(self, newID: BlockType, positions: List[VertexInfo] = None):
        self.bID = newID
        if newID is BlockType.vertex and positions is None:
            raise ValueError("A block tagged as 'vertex' MUST have at least one position passed through.")
        self.positions = positions
