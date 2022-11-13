import matplotlib.pyplot as plt
import numpy as np
import tarfile
import os


def main():
    with tarfile.open('./datasets/reconstruction/arq_dataset.tar.gz') as file:
        file.extractall('./datasets/reconstruction/')
        file.close()
    dataset = np.load('./datasets/reconstruction/custom_arq_dataset.npy', allow_pickle=True).item()
    dtest = dataset['test']

    os.remove('./datasets/reconstruction/custom_arq_dataset.npy')

    rIdx = np.random.randint(0, len(dtest['data'])-1)

    obj = {
        'label': dtest['labels'][rIdx],
        'data': np.array(dtest['data'][rIdx])
    }

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title(obj['label'])
    ax.voxels(obj['data'], facecolors='r')
    plt.show()


if __name__ == '__main__':
    main()

