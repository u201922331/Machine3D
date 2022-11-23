import tarfile
import numpy as np
import matplotlib.pyplot as plt
from utils.data_prep import get_fractured

if __name__ == '__main__':
    with tarfile.open('./datasets/arq_dataset.tar.gz') as file:
        file.extractall('./datasets/')
        file.close()
    dataset = np.load('./datasets/custom_arq_dataset.npy', allow_pickle=True).item()

    mask_tr = np.array(dataset['train']['labels']) == 'arq'
    mask_ts = np.array(dataset['test']['labels']) == 'arq'

    model_tr = dataset['train']['data'][mask_tr]
    model_ts = dataset['test']['data'][mask_ts]

    train_x, train_y = [], []
    for model in model_tr:
        fract = get_fractured(model)
        train_x.append(fract)
        train_y.append(model & ~fract)
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    test_x, test_y = [], []
    for model in model_ts:
        fract = get_fractured(model)
        test_x.append(fract)
        test_y.append(model & ~fract)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    np.save('./datasets/dataset.npy', {
        'train': {
            'x': train_x,
            'y': train_y
        },
        'test': {
            'x': test_x,
            'y': test_y
        }
    })

    index = np.random.randint(0, len(train_x))
    mesh_fract = train_x[index]
    mesh_piece = train_y[index]

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(mesh_fract, facecolors='slategray', alpha=0.5)
    ax.voxels(mesh_piece, facecolors='orange', edgecolors='darkorange')
    plt.show()
