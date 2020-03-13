from functools import partial

import torch
import torchvision
from torch import nn
from torchvision.datasets import (EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST)
from torch.optim import Optimizer
from torch.utils.data import Dataset

from .task import Task
from .mnist import Mnist
from ..models import hypernet, lenet5, mlp
from ..utils.data import split, random_split, stratified_split
from ..hyperparameters import ContiniousHyperparameter, DiscreteHyperparameter, Hyperparameters
from ..loss import F1, Accuracy, CategoricalCrossEntropy
from ..dataset import Datasets

class EMnist(Mnist):
    """
    The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19 (https://www.nist.gov/srd/nist-special-database-19)
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset (http://yann.lecun.com/exdb/mnist/).\n
    
    Paper: https://arxiv.org/abs/1702.05373v1\n
    
    Splits:\n
    'byclass', 62 classes, 697,932 train samples, 116,323 test samples, No validation, 814,255 total\n
    'bymerge', 47 classes, 697,932 train samples, 116,323 test samples, No validation, 814,255 total\n
    'balanced', 47 classes, 112,800 train samples, 18,800 test samples, Validation, 131,600 total\n
    'digits', 10 classes, 240,000 train samples, 40,000 test samples, Validation, 280,000 total\n
    'letters', 26 classes, 124,800 train samples, 20800 test samples, Validation, 145600 total\n
    'mnist', 10 classes, 60,000 train samples, 10,000 test samples, Validation, 70,000 total\n

    The subsets that are marked with 'validation' has a balanced validation set built into the training set.
    The validation set is the last portion of the training set, equal to the size of the testing set.
    """
    def __init__(self, model : str = 'lenet5_dropout', split : str = 'mnist'):
        super().__init__(model)
        self.num_classes_dict = {'byclass': 62, 'bymerge': 47, 'balanced': 47, 'letters': 26, 'digits': 10, 'mnist': 10}
        self.split = split

    @property
    def num_classes(self) -> int:
        return self.num_classes_dict[self.split]

    @property
    def model_class(self) -> hypernet.HyperNet:
        if self.model == 'lenet5_dropout':
            return partial(lenet5.Lenet5WithDropout, self.num_classes_dict[self.split])
        elif self.model == 'lenet5':
            return partial(lenet5.LeNet5, self.num_classes_dict[self.split])
        elif self.model == 'mlp':
            return partial(mlp.MLP, self.num_classes_dict[self.split])
        else:
            raise NotImplementedError

    @property
    def datasets(self) -> Datasets:
        train_data_path = test_data_path = './data'
        train_data = EMNIST(
            train_data_path,
            split=self.split,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                nn.ZeroPad2d(2),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_data = EMNIST(
            test_data_path,
            split=self.split,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                nn.ZeroPad2d(2),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        split_method = {
            'byclass': partial(stratified_split, train_data, train_data.targets, (697932-116323)/697932, 1),
            'bymerge': partial(stratified_split, train_data, train_data.targets, (697932-116323)/697932, 1),
            'balanced': partial(split, train_data, (112800-18800)/112800),
            'digits': partial(split, train_data, (240000-40000)/240000),
            'letters': partial(split, train_data, (124800-20800)/124800),
            'mnist': partial(split, train_data, (60000-10000)/60000)
            }
        if self.split in ['byclass', 'bymerge']:
            train_data, _, eval_data, _ = split_method[self.split]()
        else:
            train_data, eval_data = split_method[self.split]()
        return Datasets(train_data, eval_data, test_data)