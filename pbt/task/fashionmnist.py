from functools import partial

import torch
import torchvision
from torchvision.datasets import (EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST)
from torch.optim import Optimizer
from torch.utils.data import Dataset

from .task import Task
from ..models import hypernet, lenet5, mlp
from ..utils.data import split, random_split, stratified_split
from ..hyperparameters import ContiniousHyperparameter, DiscreteHyperparameter, Hyperparameters
from ..loss import F1, Accuracy, CategoricalCrossEntropy
from ..dataset import Datasets

class FashionMnist(Task):
    def __init__(self, model : str = 'default'):
        self.model = model
        pass
    
    @property
    def model_class(self) -> hypernet.HyperNet:
        if self.model == 'default':
            return partial(lenet5.MnistNetLarger, 10)
        elif self.model == 'lenet5':
            return lenet5.LeNet5
        elif self.model == 'mlp':
            return mlp.MLP
        else:
            raise NotImplementedError

    @property
    def optimizer_class(self) -> Optimizer:
        return torch.optim.SGD

    @property
    def hyper_parameters(self) -> Hyperparameters:
        return Hyperparameters(
            model= self.model_class.create_hyper_parameters(),
            optimizer={
                'lr': ContiniousHyperparameter(1e-9, 1e-1),
                'momentum': ContiniousHyperparameter(1e-9, 1.0),
                'weight_decay': ContiniousHyperparameter(1e-9, 1e-1),
                'nesterov': DiscreteHyperparameter(False, True)
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
        train_data, eval_data = random_split(
            train_data, fraction=54000/60000, random_state=1)
        #train_data, _, eval_data, _ = stratified_split(
        #    train_data, labels=train_data.targets, fraction=54000/60000, random_state=1)
        return Datasets(train_data, eval_data, test_data)

class FashionMnistKnowledgeSharing(FashionMnist):
    def __init__(self, model = 'lenet5'):
        self.model = model
        pass
    
    @property
    def model_class(self) -> hypernet.HyperNet:
        if self.model == 'lenet5':
            return lenet5.LeNet5
        elif self.model == 'mlp':
            return mlp.MLP
        else:
            raise NotImplementedError

    @property
    def hyper_parameters(self) -> Hyperparameters:
        return Hyperparameters(
            model= self.model_class.create_hyper_parameters(),
            optimizer={
                'lr': ContiniousHyperparameter(0.0, 1e-1),
                'momentum': ContiniousHyperparameter(0.0, 1.0)
            })

    @property
    def datasets(self) -> Datasets:
        train_data_path = test_data_path = './data'
        train_data = MNIST(
            train_data_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Pad(padding=2, padding_mode='edge'),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_data = MNIST(
            test_data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Pad(padding=2, padding_mode='edge'),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        train_data, _, eval_data, _ = stratified_split(
            train_data, labels=train_data.targets, fraction=54000/60000, random_state=1)
        return Datasets(train_data, eval_data, test_data)