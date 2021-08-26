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
        train = tonic.datasets.POKERDVS(save_to='../data/POKERDVS',train=True,download=False,transform=transform)
        test = tonic.datasets.POKERDVS(save_to='../data/POKERDVS',train=False,download=False,transform=transform)
        trainset = []
        testset = []
        trainset.append(train[0])
        testset.append(test[1])
        #trainset.append(train[1])
        #testset.append(test[1])
        """for i in range(len(train)):
            arr = []
            arr.append(train[i])
            save_arr(arr,"train_"+str(i))
        for i in range(len(test)):
            arr = []
            arr.append(test[i])
            save_arr(arr,"test_"+str(i))"""
        x_train, y_train = convert(trainset,"./arrays/train_0.npy")
        x_test, y_test = convert(testset,"./arrays/train_1.npy")
        x_train = x_test + x_train
        y_train = np.concatenate((y_train,y_test))
        x_train, x_test, y_train, y_test = np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
        y_train = npb.categorical_to_onehot(y_train, 4).astype(np.float32)
        y_test = npb.categorical_to_onehot(y_test, 4).astype(np.float32)
        print(x_train.shape)
        return x_train, y_train, x_test, y_test


def convert(set, name = None):
    x = []
    y = []
    if name == None:
        for i in range(len(set)):
            x1 = []
            y.append(set[i][1])
            for j in range(len(set[i][0])//1000):
                temp = []
                print(j)
                for k in range(len(set[i][0][0][0])):
                    for l in range(len(set[i][0][0][0][0])):
                        if set[i][0][j][0][k][l] == 1:
                            temp.append(-100)
                        elif set[i][0][j][1][k][l] == 1:
                            temp.append(100)
                        else:
                            temp.append(0)
                temp = np.array(temp)
                x1.append(temp)
            x1 = np.array(x1)
            x.append(x1)
        x = np.array(x)
    else:
        x = np.load(name)
        c = 0
        ret = []
        while c < len(x[0])-500:
            temp = []
            for i in range(500):
                temp.append(x[0][i+c])
            c += 500
            y.append(2)
            #temp = torch.FloatTensor(temp)
            ret.append(temp)
    x = ret
    y = np.array(y)
    return x, y















def save_arr(set,name):
    x = []
    y = []
    for i in range(len(set)):
        x1 = []
        y.append(set[i][1])
        for j in range(len(set[i][0])):
            temp = []
            print(j)
            for k in range(len(set[i][0][0][0])):
                for l in range(len(set[i][0][0][0][0])):
                    if set[i][0][j][0][k][l] == 1:
                        temp.append(-10)
                    elif set[i][0][j][1][k][l] == 1:
                        temp.append(10)
                    else:
                        temp.append(0)
            temp = np.array(temp)
            x1.append(temp)
        x1 = np.array(x1)
        x.append(x1)
    x = np.array(x)
    np.save("./arrays/"+name+".npy",x)

def load():
    print(np.load("./arrays/train_0.npy"))
