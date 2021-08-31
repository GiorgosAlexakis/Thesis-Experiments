import os
from abc import abstractmethod
import tonic
import tonic.transforms as transforms
import numpy as np
import torch

from .backend import numpy_backend as npb
from .utils.network_utils import download_file

_dataset_dir = os.path.join(os.path.expanduser('~'), '.neko', 'datasets')


class Dataset:
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        raise NotImplementedError


class MNIST(Dataset):
    def __init__(self):
        super().__init__()
        #self.url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
        #self.filename = 'mnist.npz'

    def load(self, onehot=True):
        """Loads MNIST dataset.

        Args:
            onehot: Whether to use onehot vector as targets.

        Returns:
            A tuple (x_train, y_train, x_test, y_test)
        """


        transform = transforms.Compose([transforms.ToSparseTensor()])
        train = tonic.datasets.POKERDVS(save_to='../data/POKERDVS',train=True,download=False)#,transform=transform)
        test = tonic.datasets.POKERDVS(save_to='../data/POKERDVS',train=False,download=False)#,transform=transform)
        x_train, x_test, y_train, y_test = [], [], [], []
        for i in range(len(train)):
            c = 0
            while c+500 < len(train[i][0]):
                x_train.append(train[i][0][c:500+c].tolist())
                y_train += [train[i][1]]
                c = c + 500
        for i in range(len(test)):
            c = 0
            while c+500 < len(test[i][0]):
                x_test.append(test[i][0][c:c+500].tolist())
                y_test += [test[i][1]]
                c = c + 500
        x_train, x_test, y_train, y_test = np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
        y_train = npb.categorical_to_onehot(y_train, 4).astype(np.float32)
        y_test = npb.categorical_to_onehot(y_test, 4).astype(np.float32)
        print(x_train.shape)
        return x_train, y_train, x_test, y_test
