from My3Dimex import handlers
from My3Dimex.vectors import Vec3
from My3Dimex.triangle import Triangle
import pandas as pd

import SimpleVoxel as svx
import numpy as np


def main():
    dataset = dict(np.load('datasets/custom_arq_dataset.npy', allow_pickle=True).tolist())
    print(dataset.keys())
    print(dataset.values())

    model = handlers.STL.read('models/suzanne.stl')
    boundaries = model.boundingBox()
    xlims = boundaries[:2]
    ylims = boundaries[2:4]
    zlims = boundaries[4:]
    w = xlims[1] - xlims[0]
    h = ylims[1] - ylims[0]
    d = zlims[1] - zlims[0]

    chunk = svx.chunk.Chunk(round(w * 20), round(h * 20), round(d * 20))
    chunk.plot()


if __name__ == '__main__':
    main()

