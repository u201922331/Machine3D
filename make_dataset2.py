import tarfile
import numpy as np


def main():
    with tarfile.open('./datasets/arq_dataset.tar.gz') as file:
        file.extractall('./datasets/')
        file.close()
    dataset = np.load('./datasets/custom_arq_dataset.npy', allow_pickle=True).item()

    mask_tr = np.array(dataset['train']['labels']) == 'arq'
    mask_ts = np.array(dataset['test']['labels']) == 'arq'

    np.save('./datasets/dataset_arq.npy', {
        'train': {
            'labels': np.array(dataset['train']['labels'])[mask_tr],
            'data': dataset['train']['data'][mask_tr],
            'errors': dataset['train']['errors']
        },
        'test': {
            'labels': np.array(dataset['test']['labels'])[mask_ts],
            'data': dataset['test']['data'][mask_ts],
            'errors': dataset['test']['errors']
        }
    })


if __name__ == '__main__':
    main()
