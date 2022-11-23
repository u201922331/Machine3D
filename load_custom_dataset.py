from os import listdir
from os.path import isfile, join
import numpy as np
from utils.constants import *
from utils.binvox_ops import *
from utils.data_prep import get_fractured


if __name__ == '__main__':
    path = PATH_MESHES+'/'
    filenames = [f for f in listdir(path) if isfile(join(path, f))]

    for name in filenames:
        if name.split('.')[-1] in ('ug', 'obj', 'off', 'dfx', 'xgl', 'pov', 'brep', 'ply', 'jot'):
            export_binvox(path+name)

    filenames = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'binvox']

    my_dataset = {'data': []}
    for name in filenames:
        my_dataset['data'].append(get_fractured(binvox_to_3d_array(path+name)))

    np.save('./datasets/my_dataset.npy', my_dataset)
