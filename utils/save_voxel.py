import trimesh.voxel.ops
import trimesh.repair
import trimesh.exchange.export


def save_voxel(path, mtx, scale=1.0):
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(mtx)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.apply_scale(scale)

    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)

    trimesh.exchange.export.export_mesh(mesh, path, path.split('.')[-1])
