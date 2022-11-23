import matplotlib.pyplot as plt


def plot_epoch(data_in, data_fake, data_target, c1 = 'gray', c2='red'):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax1.set_title('Input')
    ax2.set_title('Fake')

    ax1.voxels(data_in > 0, facecolors=c1)
    ax2.voxels(data_fake > 0, facecolors=c2)
    plt.show()
