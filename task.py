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
from hyperparameters import Hyperparameter, Hyperparameters
from loss import F1, NLL, Accuracy, BinaryCrossEntropy, CategoricalCrossEntropy
from functools import partial
from utils.data import AdaptiveDataset

@dataclass
class Task(object):
    name: str
    model_class : models.HyperNet
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
        classes = 10
        model_class = models.MnistNet10Larger
        optimizer_class = torch.optim.SGD
        loss_metric = 'cce'
        eval_metric = 'acc'
        loss_functions = {
            'cce': CategoricalCrossEntropy(),
            'acc': Accuracy(),
            'f1': F1(classes=classes)
        }
        # define hyper-parameter search space
        hyper_parameters = Hyperparameters(
            augment_params=utils.data.AdaptiveDataset.create_hyper_parameters(['degrees']),
            model_params=model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-6, 0.5),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'nesterov': Hyperparameter(False, True, is_categorical=True)
            })
        # prepare training and testing data
        train_data_path = test_data_path = './data'
        train_data = MNIST(
            train_data_path,
            train=True,
            download=True)
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
        train_data.suffix_transform = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        eval_data.suffix_transform = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        # initialize task
        super().__init__("mnist", model_class, optimizer_class, loss_metric, eval_metric, loss_functions, train_data, eval_data, test_data, hyper_parameters)

class FashionMnist(Task):
    def __init__(self):
        classes = 10
        model_class = models.MnistNet10Larger
        optimizer_class = torch.optim.SGD
        loss_metric = 'cce'
        eval_metric = 'acc'
        loss_functions = {
            'cce': CategoricalCrossEntropy(),
            'acc': Accuracy(),
            'f1': F1(classes=classes)
        }
        # define hyper-parameter search space
        hyper_parameters = Hyperparameters(
            augment_params={
                'brightness' : Hyperparameter(0.0, 1.0),
                'contrast' : Hyperparameter(0.0, 1.0),
                'saturation' : Hyperparameter(0.0, 1.0),
                'hue' : Hyperparameter(0.0, 1.0),
                'degrees' : Hyperparameter(0, 180),
                'translate_horizontal' : Hyperparameter(0.0, 1.0),
                'translate_vertical' : Hyperparameter(0.0, 1.0),
                'scale_min' : Hyperparameter(0.5, 2.0),
                'scale_max' : Hyperparameter(0.5, 2.0),
                'shear' : Hyperparameter(0, 90),
                'perspective' : Hyperparameter(0.0, 1.0),
                'vertical_flip' : Hyperparameter(0.0, 1.0),
                'horizontal_flip' : Hyperparameter(0.0, 1.0)
            },
            model_params=model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-6, 0.5),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'nesterov': Hyperparameter(False, True, is_categorical=True)
            })
        # prepare training and testing data
        train_data_path = test_data_path = './data'
        train_data = FashionMNIST(
            train_data_path,
            train=True,
            download=True)
        train_data = AdaptiveDataset(
            dataset=train_data,
            suffix_transform = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))])
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
            train_data, labels=train_data.targets, fraction=50000/60000, random_state=1)
        # initialize task
        super().__init__("fashionmnist", model_class, optimizer_class, loss_metric, eval_metric, loss_functions, train_data, eval_data, test_data, hyper_parameters)


class EMnist(Task):
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
        num_classes = {'byclass': 62, 'bymerge': 47, 'balanced': 47, 'letters': 26, 'digits': 10, 'mnist': 10}
        models_classes = {
            'byclass': models.MnistNet62Larger,
            'bymerge': models.MnistNet47Larger,
            'balanced': models.MnistNet47Larger,
            'letters': models.MnistNet26Larger,
            'digits': models.MnistNet10Larger,
            'mnist': models.MnistNet10Larger
            }
        classes = num_classes[split]
        model_class = models_classes[split]
        optimizer_class = torch.optim.SGD
        loss_metric = 'cce'
        eval_metric = 'acc'
        loss_functions = {
            'cce': CategoricalCrossEntropy(),
            'acc': Accuracy(),
            'f1': F1(classes=classes)
        }
        # define hyper-parameter search space
        hyper_parameters = Hyperparameters(
            augment_params={
                'brightness' : Hyperparameter(0.0, 1.0),
                'contrast' : Hyperparameter(0.0, 1.0),
                'saturation' : Hyperparameter(0.0, 1.0),
                'hue' : Hyperparameter(0.0, 1.0),
                #'degrees' : Hyperparameter(0, 90),
                #'translate_horizontal' : Hyperparameter(0.0, 1.0),
                #'translate_vertical' : Hyperparameter(0.0, 1.0),
                #'scale_min' : Hyperparameter(0.5, 1.5),
                #'scale_max' : Hyperparameter(0.5, 1.5),
                #'shear' : Hyperparameter(0, 90),
                #'perspective' : Hyperparameter(0.0, 1.0),
                #'vertical_flip' : Hyperparameter(0.0, 1.0),
                #'horizontal_flip' : Hyperparameter(0.0, 1.0)
            },
            model_params=model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-6, 0.5),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'nesterov': Hyperparameter(False, True, is_categorical=True)
            })
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
        split_method = {
            'byclass': partial(utils.data.stratified_split, train_data, train_data.targets, (697932-116323)/697932, 1),
            'bymerge': partial(utils.data.stratified_split, train_data, train_data.targets, (697932-116323)/697932, 1),
            'balanced': partial(utils.data.split, train_data, (112800-18800)/112800),
            'digits': partial(utils.data.split, train_data, (240000-40000)/240000),
            'letters': partial(utils.data.split, train_data, (124800-20800)/124800),
            'mnist': partial(utils.data.split, train_data, (60000-10000)/60000)
            }
        if split in ['byclass', 'bymerge']:
            train_data, _, eval_data, _ = split_method[split]()
        else:
            train_data, eval_data = split_method[split]()
        # initialize task
        super().__init__(f"emnist_{split}", model_class, optimizer_class, loss_metric, eval_metric, loss_functions, train_data, eval_data, test_data, hyper_parameters)

class CreditCardFraud(Task):
    def __init__(self):
        model_class = models.FraudNet
        optimizer_class = torch.optim.SGD
        loss_metric = 'bce'
        eval_metric = 'bce'
        loss_functions = {
            'bce': BinaryCrossEntropy(),
            'acc': Accuracy(),
            'f1': F1(classes=2)
        }
        # define hyper-parameter search space
        hyper_parameters = Hyperparameters(
            augment_params=None,
            model_params=model_class.create_hyper_parameters(),
            optimizer_params={
                'lr': Hyperparameter(1e-6, 1e-1),
                'momentum': Hyperparameter(1e-6, 0.5),
                'weight_decay': Hyperparameter(0.0, 1e-5),
                'nesterov': Hyperparameter(False, True, is_categorical=True)
            })
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
        train_data, train_labels, test_data, _ = utils.data.stratified_split(
            dataset, labels, fraction=0.9, random_state=1)
        train_data, _, eval_data, _ = utils.data.stratified_split(
            train_data, train_labels, fraction=0.9, random_state=1)
        super().__init__("creditfraud", model_class, optimizer_class, loss_metric, eval_metric, loss_functions, train_data, eval_data, test_data, hyper_parameters)
