from functools import partial

import torch
import torchvision
from torch import nn
from torchvision.datasets import CIFAR10, CIFAR100
from torch.optim import Optimizer
from torch.utils.data import Dataset

from .task import Task
from ..models import hypernet, mlp, lenet5, vgg, resnet
from ..utils.data import split, random_split, stratified_split
from ..hyperparameters import ContiniousHyperparameter, DiscreteHyperparameter, Hyperparameters
from ..loss import Accuracy, CategoricalCrossEntropy
from ..dataset import Datasets

class Cifar10(Task):
    def __init__(self, model : str = 'VGG16'):
        super().__init__()
        self.model = model

    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def model_class(self) -> hypernet.HyperNet:
        if self.model == 'lenet5':
            return partial(lenet5.LeNet5, self.num_classes)
        elif self.model == 'mlp':
            return partial(mlp.MLP, self.num_classes)
        elif self.model == 'vgg16':
            return partial(vgg.VGG16, self.num_classes, 3)
        elif self.model == 'resnet18':
            return partial(resnet.ResNet18, self.num_classes, 3)
        else:
            raise NotImplementedError

    @property
    def optimizer_class(self) -> Optimizer:
        return torch.optim.SGD

    @property
    def hyper_parameters(self) -> Hyperparameters:
        return Hyperparameters(
            model= None,
            optimizer={
                'lr': ContiniousHyperparameter(0.0, 1e-1),
                'momentum': ContiniousHyperparameter(0.0, 1.0),
                'weight_decay': ContiniousHyperparameter(0.0, 1e-2)
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
        train_data = CIFAR10(
            train_data_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]))
        test_data = CIFAR10(
            test_data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]))
        # split training set into training set and validation set
        train_data, eval_data = stratified_split(train_data, labels=train_data.targets, fraction=40000/50000, random_state=1)
        return Datasets(train_data, eval_data, test_data)

class Cifar100(Cifar10):
    def __init__(self, vgg_name = 'VGG16'):
        super().__init__(vgg_name)

    @property
    def num_classes(self) -> int:
        return 100

    @property
    def datasets(self) -> Datasets:
        train_data_path = test_data_path = './data'
        train_data = CIFAR100(
            train_data_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ]))
        test_data = CIFAR100(
            test_data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ]))
        # split training set into training set and validation set
        train_data, eval_data = stratified_split(train_data, labels=train_data.targets, fraction=40000/50000, random_state=1)
        return Datasets(train_data, eval_data, test_data)