import numpy as np
from utils.save_voxel import save_voxel
from utils.constants import *


if __name__ == '__main__':
    original_dataset_result = np.load(DATASETS+'ready.npy', allow_pickle=True).item()
    save_voxel(MESHES+'ready_broken.stl', original_dataset_result['broken'])
    save_voxel(MESHES+'ready_result.stl', original_dataset_result['result'])

    custom_dataset_result = np.load(DATASETS+'ready_custom.npy', allow_pickle=True).item()
    save_voxel(MESHES+'c_ready_broken.stl', custom_dataset_result['broken'])
    save_voxel(MESHES+'c_ready_result.stl', custom_dataset_result['result'])
