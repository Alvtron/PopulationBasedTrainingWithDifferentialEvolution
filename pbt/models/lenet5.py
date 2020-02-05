import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod, abstractstaticmethod
from .hypernet import HyperNet, Print
from ..hyperparameters import ContiniousHyperparameter

class LeNet5(HyperNet):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.modules.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)

class MnistNet(HyperNet):
    def __init__(self, output_size : int):
        super(MnistNet, self).__init__()
        self.conv_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.prelu_1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout_1 = nn.Dropout2d()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        self.prelu_2 = nn.PReLU()
        self.flatten = nn.modules.Flatten()
        self.fc_1 = nn.Linear(320, 50)
        self.prelu_3 = nn.PReLU()
        self.dropout_2 = nn.Dropout()
        self.fc_2 = nn.Linear(50, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    @staticmethod  
    def create_hyper_parameters():
        return \
        {
            'dropout_rate_1': ContiniousHyperparameter(0.0, 1.0),
            'dropout_rate_2': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_1': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_2': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_3': ContiniousHyperparameter(0.0, 1.0),
        }

    def apply_hyper_parameters(self, hyper_parameters, device):
        self.dropout_1.p = hyper_parameters['dropout_rate_1'].value
        self.dropout_2.p = hyper_parameters['dropout_rate_2'].value
        self.prelu_1 = nn.PReLU(init=hyper_parameters['prelu_alpha_1'].value).to(device)
        self.prelu_2 = nn.PReLU(init=hyper_parameters['prelu_alpha_2'].value).to(device)
        self.prelu_3 = nn.PReLU(init=hyper_parameters['prelu_alpha_3'].value).to(device)

class MnistNetLarger(HyperNet):
    def __init__(self, output_size):
        super(MnistNetLarger, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=5)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.prelu_1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(32, 128, kernel_size=5)
        self.dropout_1 = nn.Dropout2d()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        self.prelu_2 = nn.PReLU()
        self.flatten = nn.modules.Flatten()
        self.fc_1 = nn.Linear(2048, 1024)
        self.prelu_3 = nn.PReLU()
        self.dropout_2 = nn.Dropout()
        self.fc_2 = nn.Linear(1024, 512)
        self.prelu_4 = nn.PReLU()
        self.dropout_3 = nn.Dropout()
        self.fc_3 = nn.Linear(512, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    @staticmethod
    def create_hyper_parameters(include : list = None):
        hparams = {
            'dropout_rate_1': ContiniousHyperparameter(0.0, 1.0),
            'dropout_rate_2': ContiniousHyperparameter(0.0, 1.0),
            'dropout_rate_3': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_1': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_2': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_3': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_4': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_5': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_6': ContiniousHyperparameter(0.0, 1.0)}
        if not include:
            return hparams
        return {hp_name: hp_value for hp_name, hp_value in hparams.items() if hp_name in include}

    def apply_hyper_parameters(self, hyper_parameters, device):
        if 'dropout_rate_1' in hyper_parameters:
            self.dropout_1.p = hyper_parameters['dropout_rate_1'].value
        if 'dropout_rate_2' in hyper_parameters:
            self.dropout_2.p = hyper_parameters['dropout_rate_2'].value
        if 'dropout_rate_3' in hyper_parameters:
            self.dropout_3.p = hyper_parameters['dropout_rate_3'].value
        if 'prelu_alpha_1' in hyper_parameters:
            self.prelu_1 = nn.PReLU(init=hyper_parameters['prelu_alpha_1'].value).to(device)
        if 'prelu_alpha_2' in hyper_parameters:
            self.prelu_2 = nn.PReLU(init=hyper_parameters['prelu_alpha_2'].value).to(device)
        if 'prelu_alpha_3' in hyper_parameters:
            self.prelu_3 = nn.PReLU(init=hyper_parameters['prelu_alpha_3'].value).to(device)
        if 'prelu_alpha_4' in hyper_parameters:
            self.prelu_4 = nn.PReLU(init=hyper_parameters['prelu_alpha_4'].value).to(device)

class MnistNetLargest(HyperNet):
    def __init__(self, output_size : int):
        super(MnistNetLargest, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.prelu_1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.prelu_2 = nn.PReLU()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(64)
        self.prelu_3 = nn.PReLU()
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm_4 = nn.BatchNorm2d(64)
        self.prelu_4 = nn.PReLU()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.modules.Flatten()
        self.dropout_1 = nn.Dropout(p = 0.5)
        self.fc_1 = nn.Linear(64 * 7 * 7, 512)
        self.batch_norm_5 = nn.BatchNorm1d(512)
        self.prelu_5 = nn.PReLU()
        self.dropout_2 = nn.Dropout(p = 0.5)
        self.fc_2 = nn.Linear(512, 512)
        self.batch_norm_6 = nn.BatchNorm1d(512)
        self.prelu_6 = nn.PReLU()
        self.dropout_3 = nn.Dropout(p = 0.5)
        self.fc_3 = nn.Linear(512, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    @staticmethod
    def create_hyper_parameters():
        return \
        {
            'dropout_rate_1': ContiniousHyperparameter(0.0, 1.0),
            'dropout_rate_2': ContiniousHyperparameter(0.0, 1.0),
            'dropout_rate_3': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_1': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_2': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_3': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_4': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_5': ContiniousHyperparameter(0.0, 1.0),
            'prelu_alpha_6': ContiniousHyperparameter(0.0, 1.0)
        }

    def apply_hyper_parameters(self, hyper_parameters, device):
        self.dropout_1.p = hyper_parameters['dropout_rate_1'].value
        self.dropout_2.p = hyper_parameters['dropout_rate_2'].value
        self.dropout_3.p = hyper_parameters['dropout_rate_3'].value
        self.prelu_1 = nn.PReLU(init=hyper_parameters['prelu_alpha_1'].value).to(device)
        self.prelu_2 = nn.PReLU(init=hyper_parameters['prelu_alpha_2'].value).to(device)
        self.prelu_3 = nn.PReLU(init=hyper_parameters['prelu_alpha_3'].value).to(device)
        self.prelu_4 = nn.PReLU(init=hyper_parameters['prelu_alpha_4'].value).to(device)
        self.prelu_5 = nn.PReLU(init=hyper_parameters['prelu_alpha_5'].value).to(device)
        self.prelu_6 = nn.PReLU(init=hyper_parameters['prelu_alpha_6'].value).to(device)