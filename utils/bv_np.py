import numpy as np


class Voxels(object):
    def __init__(self, data, dims, t, s, axis_order):
        self.data = data
        self.dims = dims
        self.translate = t
        self.scale = s
        assert(axis_order in ('xyz', 'xzy'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)


def read_header(handler):
    line = handler.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, handler.readline().strip().split(b' ')[1:]))
    translate = list(map(float, handler.readline().strip().split(b' ')[1:]))
    scale = list(map(float, handler.readline().strip().split(b' ')[1:]))[0]
    line = handler.readline()
    return dims, translate, scale


def read_as_3d_array(path, fix_coords=False):
    with open(path, 'rb') as f:
        dims, translate, scale = read_header(f)
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)

        values, counts = raw_data[::2], raw_data[1::2]
        data = np.repeat(values, counts).astype(np.bool)
        data = data.reshape(dims)
        if fix_coords:
            data = np.transpose(data, (0, 2, 1))
            axis_order = 'xyz'
        else:
            axis_order = 'xzy'
        return Voxels(data, dims, translate, scale, axis_order)
