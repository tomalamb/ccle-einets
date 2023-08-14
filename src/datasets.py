"""
Code for loading data from datasets. Code builds upon the code from the original 
EiNets repository, which can be found at https://github.com/cambridge-mlg/EinsumNetworks.
"""

import numpy as np
import os
import tempfile
import urllib.request
import utils as utils
import shutil
import gzip
import scipy.io as sp
from enum import Enum
from typing import Tuple


class Dataset(Enum):
    # Create dataset class for loading the different datasets.
    MNIST = 1
    F_MNIST = 2


def maybe_download(directory, url_base, filename):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        return False

    if not os.path.isdir(directory):
        utils.mkdir_p(directory)

    url = url_base + filename
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading {} to {}'.format(url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    print('{} Bytes'.format(os.path.getsize(zipped_filepath)))
    print('Move to {}'.format(filepath))
    shutil.move(zipped_filepath, filepath)
    return True


def maybe_download_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        if not maybe_download('../data/datasets/mnist', 'http://yann.lecun.com/exdb/mnist/', file):
            continue
        print('unzip ../data/datasets/mnist/{}'.format(file))
        filepath = os.path.join('../data/datasets/mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_mnist(path):
    """Load MNIST dataset.

    Args:
        path (str): Gives path to directory containing MNIST dataset. Variable
        so can adapt for use on cluster.

    Returns:
        : _description_
    """

    maybe_download_mnist()

    if path == None:
        path = '../data/datasets/mnist'
    else:
        data_dir = path

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_x, train_labels, test_x, test_labels


def maybe_download_fashion_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        if not maybe_download('../data/datasets/f_mnist', 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', file):
            continue
        print('unzip ../data/datasets/f_mnist/{}'.format(file))
        filepath = os.path.join('../data/datasets/f_mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_fashion_mnist(path):
    """Load fashion-MNIST"""

    maybe_download_fashion_mnist()

    if path == None:
        path = '../data/datasets/f_mnist'
    else:
        data_dir = path

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_x, train_labels, test_x, test_labels


def load_data(data_set: Dataset, path=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load specified dataset.

    Args:
        data_set (Dataset): dataset of type Dataset to load.

    Raises:
        ValueError: If dataset is not known, i.e. not specified in the Dataset class.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: train_x, train_labels, test_x, test_labels
        of the dataset specified.
    """
    if data_set is Dataset.MNIST:
        return load_mnist(path)
    elif data_set is Dataset.F_MNIST:
        return load_fashion_mnist(path)
    else:
        raise ValueError(
            "Currently only supporting MNIST and F-MNSIT datasets.")


if __name__ == '__main__':
    print('Downloading datasets -- this might take a while')

    print()
    print('MNIST')
    maybe_download_mnist()

    print()
    print('fashion MNIST')
    maybe_download_fashion_mnist()
