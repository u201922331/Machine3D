from My3Dimex import handlers
from My3Dimex.vectors import Vec3
from My3Dimex.triangle import Triangle
import pandas as pd

import SimpleVoxel as svx


def main():
    cube = handlers.STL.read('models/suzanne.stl')
    boundaries = cube.boundingBox()
    xlims = boundaries[:2]
    ylims = boundaries[2:4]
    zlims = boundaries[4:]
    w = xlims[1]-xlims[0]
    h = ylims[1]-ylims[0]
    d = zlims[1]-zlims[0]

    chunk = svx.chunk.Chunk(round(w*10), round(h*10), round(d*10))
    chunk.plot()


if __name__ == '__main__':
    main()

