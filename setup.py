import os
import sys
import argparse
import math
import torch
import torchvision
import torch.utils.data
import pandas
import numpy
import random
import sklearn.preprocessing
import sklearn.model_selection
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST, ImageNet, CIFAR10, CIFAR100
from models import MnistNet1, MnistNet2, MnistNet3, FraudNet
from database import SharedDatabase
from hyperparameters import Hyperparameter, Hyperparameters
from controller import Controller
from evaluator import Evaluator
from trainer import Trainer
from evolution import ExploitAndExplore, DifferentialEvolution, ParticleSwarm
from analyze import Analyzer
from loss import CategoricalCrossEntropy, BinaryCrossEntropy, Accuracy, F1, NLL

def split_dataset(dataset, fraction):
    assert 0.0 <= fraction <= 1.0, f"The provided fraction must be between 0.0 and 1.0!"
    dataset_length = len(dataset)
    first_set_length = math.floor(fraction * dataset_length)
    second_set_length = dataset_length - first_set_length
    first_set, second_set = torch.utils.data.random_split(dataset, (first_set_length, second_set_length))
    return first_set, second_set

def setup_mnist():
    model_class = MnistNet2
    optimizer_class = torch.optim.SGD
    loss_metric = 'nll'
    eval_metric = 'acc'
    eval_metrics = {
        #'cce': CategoricalCrossEntropy(),
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
        general_params = None,
        model_params = model_class.create_hyper_parameters(),
        optimizer_params = {
            'lr': Hyperparameter(1e-6, 1e-2), # Learning rate.
            'momentum': Hyperparameter(1e-1, 1e-0), # Parameter that accelerates SGD in the relevant direction and dampens oscillations.
            'weight_decay': Hyperparameter(0.0, 1e-5), # Learning rate decay over each update.
            'nesterov': Hyperparameter(False, True, is_categorical = True) # Whether to apply Nesterov momentum.
            })
    return model_class, optimizer_class, loss_metric, eval_metric, eval_metrics, train_data, eval_data, test_data, hyper_parameters

def setup_emnist():
    model_class = MnistNet2
    optimizer_class = torch.optim.Adam
    loss_metric = 'nll'
    eval_metric = 'acc'
    eval_metrics = {
        #'cce': CategoricalCrossEntropy(),
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
        general_params = None,
        model_params = model_class.create_hyper_parameters(),
        optimizer_params = {
            'lr': Hyperparameter(1e-10, 1e-2),
            'betas': Hyperparameter((0.9, 0.999), is_categorical=True),
            'eps': Hyperparameter(1e-10, 1.0),
            'weight_decay': Hyperparameter(0.0, 1e-5), # Learning rate decay over each update.
            'amsgrad': Hyperparameter(False, True, is_categorical = True)
            })
    return model_class, optimizer_class, loss_metric, eval_metric, eval_metrics, train_data, eval_data, test_data, hyper_parameters
  

def setup_fraud():
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
    X = df.iloc[:, :-1].values # extracting features
    y = df.iloc[:, -1].values # extracting labels
    sc = sklearn.preprocessing.StandardScaler()
    X = sc.fit_transform(X)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
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
        general_params = None,
        model_params = model_class.create_hyper_parameters(),
        optimizer_params = {
            'lr': Hyperparameter(1e-6, 1e-1), # Learning rate.
            'momentum': Hyperparameter(1e-1, 1e-0), # Parameter that accelerates SGD in the relevant direction and dampens oscillations.
            'weight_decay': Hyperparameter(0.0, 1e-5), # Learning rate decay over each update.
            'nesterov': Hyperparameter(False, True, is_categorical = True) # Whether to apply Nesterov momentum.
            })
    return model_class, optimizer_class, loss_metric, eval_metric, eval_metrics, train_data, eval_data, test_data, hyper_parameters
