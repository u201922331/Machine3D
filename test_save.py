import numpy as np
import matplotlib.pyplot as plt
import trimesh.voxel.ops
import trimesh.repair
import trimesh.exchange.export


if __name__ == '__main__':
    dataset = np.load('./datasets/dataset.npy', allow_pickle=True).item()

    mtr_x, mtr_y = dataset['train']['x'], dataset['train']['y']
    mts_x, mts_y = dataset['test']['x'], dataset['test']['y']

    index = np.random.randint(0, len(mtr_x)-1)
    model_fract = mtr_x[index]
    model_piece = mtr_y[index]

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(model_fract, facecolors='slategray', alpha=0.5)
    ax.voxels(model_piece, facecolors='orange', edgecolors='darkorange')
    plt.show()

    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(model_fract)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.apply_scale(0.25)
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)

    trimesh.exchange.export.export_mesh(mesh, file_obj='./meshes/test_mesh_fract.stl', file_type='stl')

    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(model_piece)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.apply_scale(0.25)
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)

    trimesh.exchange.export.export_mesh(mesh, file_obj='./meshes/test_mesh_piece.stl', file_type='stl')
