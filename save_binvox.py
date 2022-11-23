import matplotlib.pyplot as plt
import utils.binvox_ops as bvo

if __name__ == '__main__':
    bvo.export_binvox('./meshes/jarron.obj')
    voxels = bvo.binvox_to_3d_array('./meshes/jarron.binvox')

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels, facecolors='red')
    plt.show()
