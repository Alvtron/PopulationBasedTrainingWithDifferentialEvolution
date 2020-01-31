import pandas
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.utils.data
import torchvision
import models
import utils.data
import numpy as np
from torchvision.datasets import (CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST, ImageNet)
from dataclasses import dataclass
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchvision.datasets.vision import StandardTransform
from hyperparameters import Hyperparameter, Hyperparameters
from loss import F1, NLL, Accuracy, BinaryCrossEntropy, CategoricalCrossEntropy
from functools import partial
from utils.data import AdaptiveDataset
from abc import ABC, abstractmethod, abstractproperty

class Datasets(object):
    def __init__(self, train, eval, test):
        self.train = train
        self.eval = eval
        self.test = test

class Task(ABC):
    """
    Base class for all tasks.
    """

    @property
    def model_class(self) -> models.HyperNet:
        raise NotImplementedError()

    @property
    def optimizer_class(self) -> Optimizer:
        raise NotImplementedError()

    @property
    def hyper_parameters(self) -> Hyperparameters:
        """Define hyper-parameter search space. """
        raise NotImplementedError()

    @property
    def loss_functions(self) -> dict:
        raise NotImplementedError()

    @property
    def loss_metric(self) -> str:
        raise NotImplementedError()

    @property
    def eval_metric(self) -> str:
        raise NotImplementedError()

    @property
    def datasets(self) -> Datasets:
        raise NotImplementedError()

class CreditCardFraud(Task):
    def __init__(self):
        pass

    @property
    def model_class(self) -> models.HyperNet:
        return models.FraudNet

    @property
    def optimizer_class(self) -> Optimizer:
        return torch.optim.SGD

    @property
    def hyper_parameters(self) -> Hyperparameters:
        return Hyperparameters(
            augment_params=None,
            model_params= self.model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-6, 1.0),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'nesterov': Hyperparameter(False, True, is_categorical=True)
            })

    @property
    def loss_functions(self) -> dict:
        return \
        {
            'bce': BinaryCrossEntropy(),
            'acc': Accuracy(),
            'f1': F1(classes=2)
        }

    @property
    def loss_metric(self) -> str:
        return 'bce'

    @property
    def eval_metric(self) -> str:
        return 'bce'

    @property
    def datasets(self) -> Datasets:
        df = pandas.read_csv('./data/CreditCardFraud/creditcard.csv')
        inputs = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        sc = sklearn.preprocessing.StandardScaler()
        torch_inputs = sc.fit_transform(inputs)
        torch_inputs = torch.from_numpy(torch_inputs).float()
        torch_labels = torch.from_numpy(labels).float()
        dataset = torch.utils.data.TensorDataset(torch_inputs, torch_labels)
        # split dataset into training-, testing- and validation set
        train_data, train_labels, test_data, _ = utils.data.stratified_split(
            dataset, labels, fraction=0.9, random_state=1)
        train_data, _, eval_data, _ = utils.data.stratified_split(
            train_data, train_labels, fraction=0.9, random_state=1)
        return Datasets(train_data, eval_data, test_data)

class Mnist(Task):
    def __init__(self):
        pass

    @property
    def model_class(self) -> models.HyperNet:
        return models.MnistNet10Larger

    @property
    def optimizer_class(self) -> Optimizer:
        return torch.optim.SGD

    @property
    def hyper_parameters(self) -> Hyperparameters:
        return Hyperparameters(
            augment_params=None,
            model_params= self.model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-6, 1.0),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'nesterov': Hyperparameter(False, True, is_categorical=True)
            })

    @property
    def loss_functions(self) -> dict:
        return \
        {
            'cce': CategoricalCrossEntropy(),
            'acc': Accuracy()
        }

    @property
    def loss_metric(self) -> str:
        return 'cce'

    @property
    def eval_metric(self) -> str:
        return 'cce'

    @property
    def datasets(self) -> Datasets:
        train_data_path = test_data_path = './data'
        train_data = MNIST(
            train_data_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_data = MNIST(
            test_data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        train_data, _, eval_data, _ = utils.data.stratified_split(
            train_data, labels=train_data.targets, fraction=50000/60000, random_state=1)
        return Datasets(train_data, eval_data, test_data)
        
class MnistKnowledgeSharing(Mnist):
    def __init__(self):
        pass
    
    @property
    def hyper_parameters(self) -> Hyperparameters:
        model_hyper_parameters = self.model_class.create_hyper_parameters(['dropout_rate_1', 'dropout_rate_2', 'dropout_rate_3'])
        return Hyperparameters(
            augment_params=None,
            model_params= model_hyper_parameters,
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-6, 1.0)
            })

    @property
    def datasets(self) -> Datasets:
        train_data_path = test_data_path = './data'
        train_data = MNIST(
            train_data_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_data = MNIST(
            test_data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        train_data, _, eval_data, _ = utils.data.stratified_split(
            train_data, labels=train_data.targets, fraction=54000/60000, random_state=1)
        return Datasets(train_data, eval_data, test_data)

class FashionMnist(Mnist):
    def __init__(self):
        pass
    
    @property
    def datasets(self) -> Datasets:
        train_data_path = test_data_path = './data'
        train_data = FashionMNIST(
            train_data_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_data = FashionMNIST(
            test_data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        train_data, _, eval_data, _ = utils.data.stratified_split(
            train_data, labels=train_data.targets, fraction=54000/60000, random_state=1)
        return Datasets(train_data, eval_data, test_data)

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
    def __init__(self, split : str = 'mnist'):
        self.split = split
        self.num_classes = {'byclass': 62, 'bymerge': 47, 'balanced': 47, 'letters': 26, 'digits': 10, 'mnist': 10}
        self.models_classes = {
            'byclass': models.MnistNet62Larger,
            'bymerge': models.MnistNet47Larger,
            'balanced': models.MnistNet47Larger,
            'letters': models.MnistNet26Larger,
            'digits': models.MnistNet10Larger,
            'mnist': models.MnistNet10Larger}
    
    @property
    def num_classes(self) -> int:
        return self.num_classes[self.split]

    @property
    def model_class(self) -> models.HyperNet:
        return self.models_classes[self.split]

    @property
    def optimizer_class(self) -> Optimizer:
        return torch.optim.SGD

    @property
    def hyper_parameters(self) -> Hyperparameters:
        return Hyperparameters(
            augment_params=None,
            model_params=self.model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-6, 0.5),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'nesterov': Hyperparameter(False, True, is_categorical=True)
            })

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
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_data = EMNIST(
            test_data_path,
            split=self.split,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        split_method = {
            'byclass': partial(utils.data.stratified_split, train_data, train_data.targets, (697932-116323)/697932, 1),
            'bymerge': partial(utils.data.stratified_split, train_data, train_data.targets, (697932-116323)/697932, 1),
            'balanced': partial(utils.data.split, train_data, (112800-18800)/112800),
            'digits': partial(utils.data.split, train_data, (240000-40000)/240000),
            'letters': partial(utils.data.split, train_data, (124800-20800)/124800),
            'mnist': partial(utils.data.split, train_data, (60000-10000)/60000)
            }
        if self.split in ['byclass', 'bymerge']:
            train_data, _, eval_data, _ = split_method[self.split]()
        else:
            train_data, eval_data = split_method[self.split]()
        return Datasets(train_data, eval_data, test_data)