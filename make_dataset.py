import numpy as np
import matplotlib.pyplot as plt
import tarfile
import os
from data_prep import get_fractured


if __name__ == '__main__':
    # Load the original dataset
    # ==========================
    # Given that Github doesn't allow uploading bigger files, we're uncompressing
    # the original database's compressed form (which is with the tar.gz extension)
    with tarfile.open('./datasets/reconstruction/arq_dataset.tar.gz') as file:
        file.extractall('./datasets/reconstruction/')
        file.close()
    dataset = np.load('./datasets/reconstruction/custom_arq_dataset.npy', allow_pickle=True).item()
    # Since we're not needing this file anymore, we simply delete it
    os.remove('./datasets/reconstruction/custom_arq_dataset.npy')
    # Now we get back to processing the data
    mask_tr = np.array(dataset['train']['labels']) == 'arq'
    mask_ts = np.array(dataset['train']['labels']) == 'arq'

    model_tr = dataset['train']['data'][mask_tr]
    model_ts = dataset['train']['data'][mask_ts]

    # Prepare the training data
    # ==========================
    train_x, train_y = [], []
    for model in model_tr:
        fract = get_fractured(model)
        train_x.append(fract)  # Broken object
        train_y.append(model & ~fract)  # Target
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # Prepare the testing data
    # =========================
    test_x, test_y = [], []
    for model in model_ts:
        fract = get_fractured(model)
        test_x.append(fract)
        test_y.append(model & ~fract)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # Save
    np.save("./datasets/dataset.npy", {
        'train': {
            'x': train_x,
            'y': train_y
        },
        'test': {
            'x': test_x,
            'y': test_y
        }
    })

    # To test whether the fracturing worked or not, we plot it
    # =========================================================
    index = np.random.randint(0, len(train_x)-1)
    mesh_fract = train_x[index]
    mesh_piece = train_y[index]

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(mesh_fract, facecolors='slategray', alpha=.5)
    ax.voxels(mesh_piece, facecolors='orange', edgecolors='darkorange')
    plt.show()
