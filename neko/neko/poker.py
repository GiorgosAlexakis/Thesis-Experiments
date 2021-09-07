import os
from abc import abstractmethod
import tonic
import tonic.transforms as transforms
import numpy as np
import torch
from tqdm import tqdm
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


        transform = transforms.Compose([
        tonic.transforms.Downsample( time_factor=0.00001)])
        train = tonic.datasets.DVSGesture(save_to='../data/dvsgesture/',train=True,download=False)#,transform=transform)
        test = tonic.datasets.DVSGesture(save_to='../data/dvsgesture/',train=False,download=False)#,transform=transform)
        y_train, y_test = np.array([]), np.array([])
        x_train = np.array([train[0][0][0:10000:1000]])
        x_test = np.array([test[0][0][0:10000:1000]])
        for i in tqdm(range(len(train))):
            c = 0
            while c+10000 < len(train[i][0]):
                x_train = np.append(x_train,[train[i][0][c:10000+c:1000]],axis=0)
                y_train = np.append(y_train,[train[i][1]])
                c = c + 10000
        for i in tqdm(range(len(test))):
            c = 0
            while c+10000 < len(test[i][0]):
                x_test = np.append(x_test, [test[i][0][c:10000 + c:1000]],axis=0)
                y_test = np.append(y_test,[test[i][1]])
                c = c + 10000
        x_train, x_test, y_train, y_test = np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
        x_train, x_test = x_train[1:], x_test[1:]
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        y_train = npb.categorical_to_onehot(y_train, 11).astype(np.float32)
        y_test = npb.categorical_to_onehot(y_test, 11).astype(np.float32)
        return x_train, y_train, x_test, y_test
