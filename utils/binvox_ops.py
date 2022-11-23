from trimesh.exchange.binvox import Binvoxer
from .bv_np import read_as_3d_array


def export_binvox(src_path, dim=32):
    bvx = Binvoxer(dimension=dim, center=True, verbose=True, use_offscreen_pbuffer=False, exact=True)
    bvx(src_path, overwrite=True)


def binvox_to_3d_array(binvox_path):
    return read_as_3d_array(binvox_path).data
