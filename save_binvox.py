import matplotlib.pyplot as plt
import utils.binvox_ops as bvo
from utils.constants import *

if __name__ == '__main__':
    bvo.export_binvox(MESHES+'jarron.obj')
    voxels = bvo.binvox_to_3d_array(MESHES+'jarron.binvox')

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels, facecolors='red')
    plt.show()
