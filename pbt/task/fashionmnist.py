from functools import partial

import torch
import torchvision
from torch import nn
from torchvision.datasets import FashionMNIST
from torch.optim import Optimizer
from torch.utils.data import Dataset

from .task import Task
from ..models import hypernet, lenet5, mlp, vgg, resnet
from ..utils.data import split, random_split, stratified_split
from ..hyperparameters import ContiniousHyperparameter, DiscreteHyperparameter, Hyperparameters
from ..loss import F1, Accuracy, CategoricalCrossEntropy
from ..dataset import Datasets


cce = CategoricalCrossEntropy()
f1 = F1(classes=10)
accuracy = Accuracy()


class FashionMnist(Task):

    def __init__(self, model : str = 'lenet5'):
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
            return partial(vgg.VGG16, self.num_classes, 1)
        elif self.model == 'resnet18':
            return partial(resnet.ResNet18, self.num_classes, 1)
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
                'lr': ContiniousHyperparameter(1e-5, 1e-1),
                'momentum': ContiniousHyperparameter(0.8, 1.0),
                'weight_decay': ContiniousHyperparameter(0.0, 1e-3),
            })

    @property
    def loss_functions(self) -> dict:
        return {
            cce.iso: cce,
            f1.iso: f1,
            accuracy.iso: accuracy
        }

    @property
    def loss_metric(self) -> str:
        return cce.iso

    @property
    def eval_metric(self) -> str:
        return f1.iso

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