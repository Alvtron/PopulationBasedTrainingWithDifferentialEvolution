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

def split_dataset(dataset, fraction):
    assert 0.0 <= fraction <= 1.0, f"The provided fraction must be between 0.0 and 1.0!"
    dataset_length = len(dataset)
    first_set_length = math.floor(fraction * dataset_length)
    second_set_length = dataset_length - first_set_length
    first_set, second_set = torch.utils.data.random_split(
        dataset, (first_set_length, second_set_length))
    return first_set, second_set

@dataclass
class Task(object):
    name: str
    model_class : HyperNet
    optimizer_class : Optimizer
    loss_metric : str
    eval_metric : str
    eval_metrics : dict
    train_data : Dataset
    eval_data : Dataset
    test_data : Dataset
    hyper_parameters : Hyperparameters

class Mnist(Task):
    def __init__(self):
        model_class = MnistNet2
        optimizer_class = torch.optim.SGD
        loss_metric = 'nll'
        eval_metric = 'acc'
        eval_metrics = {
            # 'cce': CategoricalCrossEntropy(),
            'nll': NLL(),
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
        train_data, eval_data = split_dataset(train_data, 0.90)
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
        super().__init__("mnist", model_class, optimizer_class, loss_metric, eval_metric, eval_metrics, train_data, eval_data, test_data, hyper_parameters)


class EMnist(Task):
    def __init__(self):
        model_class = MnistNet2
        optimizer_class = torch.optim.Adam
        loss_metric = 'nll'
        eval_metric = 'acc'
        eval_metrics = {
            # 'cce': CategoricalCrossEntropy(),
            'nll': NLL(),
            'acc': Accuracy()
        }
        # prepare training and testing data
        train_data_path = test_data_path = './data'
        train_data = EMNIST(
            train_data_path,
            split='byclass',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_data = EMNIST(
            test_data_path,
            split='byclass',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        train_data, eval_data = split_dataset(train_data, 0.90)
        # define hyper-parameter search space
        hyper_parameters = Hyperparameters(
            general_params=None,
            model_params=model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-10, 1e-2),
                'betas': Hyperparameter((0.9, 0.999), is_categorical=True),
                'eps': Hyperparameter(1e-10, 1.0),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'amsgrad': Hyperparameter(False, True, is_categorical=True)
            })
        super().__init__("emnist", model_class, optimizer_class, loss_metric, eval_metric, eval_metrics, train_data, eval_data, test_data, hyper_parameters)


class Fraud(Task):
    def __init__(self):
        model_class = FraudNet
        optimizer_class = torch.optim.SGD
        loss_metric = 'bce'
        eval_metric = 'bce'
        eval_metrics = {
            'bce': BinaryCrossEntropy(),
            'acc': Accuracy()
        }
        # prepare training and testing data
        df = pandas.read_csv('./data/CreditCardFraud/creditcard.csv')
        X = df.iloc[:, :-1].values  # extracting features
        y = df.iloc[:, -1].values  # extracting labels
        sc = sklearn.preprocessing.StandardScaler()
        X = sc.fit_transform(X)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.1, random_state=1)
        X_train = torch.from_numpy(X_train).float()
        Y_train = torch.from_numpy(Y_train).float()
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test).float()
        train_data = torch.utils.data.TensorDataset(X_train, Y_train)
        test_data = torch.utils.data.TensorDataset(X_test, Y_test)
        # split training set into training set and validation set
        train_data, eval_data = split_dataset(train_data, 0.9)
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
        super().__init__("fraud", model_class, optimizer_class, loss_metric, eval_metric, eval_metrics, train_data, eval_data, test_data, hyper_parameters)
