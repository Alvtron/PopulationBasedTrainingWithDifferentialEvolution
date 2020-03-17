from functools import partial

import torch
import torchvision
from torch import nn
from torchvision.datasets import FashionMNIST
from torch.optim import Optimizer
from torch.utils.data import Dataset

from .task import Task
from ..models import hypernet, lenet5, mlp
from ..utils.data import split, random_split, stratified_split
from ..hyperparameters import ContiniousHyperparameter, DiscreteHyperparameter, Hyperparameters
from ..loss import F1, Accuracy, CategoricalCrossEntropy
from ..dataset import Datasets

class FashionMnist(Task):
    def __init__(self, model : str = 'lenet5_dropout'):
        super().__init__()
        self.model = model

    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def model_class(self) -> hypernet.HyperNet:
        if self.model == 'lenet5_dropout':
            return partial(lenet5.Lenet5WithDropout, self.num_classes)
        elif self.model == 'lenet5':
            return partial(lenet5.LeNet5, self.num_classes)
        elif self.model == 'mlp':
            return partial(mlp.MLP, self.num_classes)
        else:
            raise NotImplementedError

    @property
    def optimizer_class(self) -> Optimizer:
        return torch.optim.SGD

    @property
    def hyper_parameters(self) -> Hyperparameters:
        model_hyper_parameters = None
        if self.model == 'lenet5_dropout':
            model_hyper_parameters = lenet5.Lenet5WithDropout.create_hyper_parameters()
        return Hyperparameters(
            model= model_hyper_parameters,
            optimizer={
                'lr': ContiniousHyperparameter(1e-9, 1e-1),
                'momentum': ContiniousHyperparameter(1e-9, 1.0)
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
                nn.ZeroPad2d(2),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_data = FashionMNIST(
            test_data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                nn.ZeroPad2d(2),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        train_data, eval_data = stratified_split(train_data, labels=train_data.targets, fraction=50000/60000, random_state=1)
        return Datasets(train_data, eval_data, test_data)

class FashionMnistKnowledgeSharing(FashionMnist):
    def __init__(self, model = 'lenet5'):
        super().__init__(model)

    @property
    def hyper_parameters(self) -> Hyperparameters:
        return Hyperparameters(
            optimizer={
                'lr': ContiniousHyperparameter(0.0, 1e-1),
                'momentum': ContiniousHyperparameter(0.0, 1.0)
            })

    @property
    def datasets(self) -> Datasets:
        train_data_path = test_data_path = './data'
        train_data = FashionMNIST(
            train_data_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                nn.ZeroPad2d(2),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_data = FashionMNIST(
            test_data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                nn.ZeroPad2d(2),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        train_data, eval_data = stratified_split(train_data, labels=train_data.targets, fraction=54000/60000, random_state=1)
        return Datasets(train_data, eval_data, test_data)