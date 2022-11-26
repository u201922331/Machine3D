import numpy as np
from utils.save_voxel import save_voxel


if __name__ == '__main__':
    original_dataset_result = np.load('./datasets/ready.npy', allow_pickle=True).item()
    save_voxel('./meshes/ready_broken.stl', original_dataset_result['broken'])
    save_voxel('./meshes/ready_result.stl', original_dataset_result['result'])

    custom_dataset_result = np.load('./datasets/ready_custom.npy', allow_pickle=True).item()
    save_voxel('./meshes/c_ready_broken.stl', custom_dataset_result['broken'])
    save_voxel('./meshes/c_ready_result.stl', custom_dataset_result['result'])
