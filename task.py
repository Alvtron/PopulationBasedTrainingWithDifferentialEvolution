import math
import pandas
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.utils.data
import torchvision
from torchvision.datasets import (CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST, ImageNet)
from dataclasses import dataclass
from torch.optim import Optimizer
from torch.utils.data import Dataset
from hyperparameters import Hyperparameter, Hyperparameters
from loss import F1, NLL, Accuracy, BinaryCrossEntropy, CategoricalCrossEntropy
from models import HyperNet, FraudNet, MnistNet1, MnistNet2, MnistNet3
from utils.data import random_split, stratified_split
@dataclass
class Task(object):
    name: str
    model_class : HyperNet
    optimizer_class : Optimizer
    loss_metric : str
    eval_metric : str
    loss_functions : dict
    train_data : Dataset
    eval_data : Dataset
    test_data : Dataset
    hyper_parameters : Hyperparameters

class Mnist(Task):
    def __init__(self):
        model_class = MnistNet2
        optimizer_class = torch.optim.SGD
        loss_metric = 'cce'
        eval_metric = 'acc'
        loss_functions = {
            'cce': CategoricalCrossEntropy(),
            #'nll': NLL(),
            'acc': Accuracy()
        }
        # prepare training and testing data
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
        train_data, eval_data = random_split(train_data, 0.90)
        # define hyper-parameter search space
        hyper_parameters = Hyperparameters(
            general_params=None,
            model_params=model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-1, 1e-0),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'nesterov': Hyperparameter(False, True, is_categorical=True)
            })
        super().__init__("mnist", model_class, optimizer_class, loss_metric, eval_metric, loss_functions, train_data, eval_data, test_data, hyper_parameters)

class EMnist(Task):
    def __init__(self, split='mnist'):
        model_class = MnistNet2
        optimizer_class = torch.optim.Adam
        loss_metric = 'nll'
        eval_metric = 'acc'
        loss_functions = {
            # 'cce': CategoricalCrossEntropy(),
            'nll': NLL(),
            'acc': Accuracy()
        }
        # prepare dataset
        train_data_path = test_data_path = './data'
        train_data = EMNIST(
            train_data_path,
            split=split,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_data = EMNIST(
            test_data_path,
            split=split,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        train_data, eval_data = random_split(train_data, 0.90)
        # define hyper-parameter search space
        hyper_parameters = Hyperparameters(
            general_params=None,
            model_params=model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-1, 1e-0),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'nesterov': Hyperparameter(False, True, is_categorical=True)
            })
        super().__init__("emnist", model_class, optimizer_class, loss_metric, eval_metric, loss_functions, train_data, eval_data, test_data, hyper_parameters)

class CreditCardFraud(Task):
    def __init__(self):
        model_class = FraudNet
        optimizer_class = torch.optim.SGD
        loss_metric = 'bce'
        eval_metric = 'bce'
        loss_functions = {
            'bce': BinaryCrossEntropy(),
            'acc': Accuracy()
        }
        # prepare dataset
        df = pandas.read_csv('./data/CreditCardFraud/creditcard.csv')
        inputs = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        sc = sklearn.preprocessing.StandardScaler()
        torch_inputs = sc.fit_transform(inputs)
        torch_inputs = torch.from_numpy(torch_inputs).float()
        torch_labels = torch.from_numpy(labels).float()
        dataset = torch.utils.data.TensorDataset(torch_inputs, torch_labels)
        # split dataset into training-, testing- and validation set
        train_data, train_labels, test_data, _ = stratified_split(dataset, labels, fraction=0.9, random_state=1)
        train_data, train_labels, eval_data, _ = stratified_split(train_data, train_labels, fraction=0.9, random_state=1)
        # define hyper-parameter search space
        hyper_parameters = Hyperparameters(
            general_params=None,
            model_params=model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-1, 1e-0),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'nesterov': Hyperparameter(False, True, is_categorical=True)
            })
        super().__init__("fraud", model_class, optimizer_class, loss_metric, eval_metric, loss_functions, train_data, eval_data, test_data, hyper_parameters)
