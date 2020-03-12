import math
from abc import abstractmethod, abstractstaticmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypernet import HyperNet, Print
from ..hyperparameters import ContiniousHyperparameter

class LeNet5(HyperNet):
    def __init__(self, output_size : int):
        super().__init__()
        if not 0 < output_size <= 84:
            raise ValueError("Output size must be between 1 and 84.")
        self.padding = nn.ZeroPad2d(2)
        self.conv_1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu_2 = nn.ReLU()
        self.flatten = nn.modules.Flatten()
        self.fc_1 = nn.Linear(16*5*5, 120)
        self.relu_3 = nn.ReLU()
        self.fc_2 = nn.Linear(120, 84)
        self.relu_4 = nn.ReLU()
        self.fc_3 = nn.Linear(84, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

class Lenet5WithDropout(HyperNet):
    def __init__(self, output_size : int):
        super().__init__()
        if not 0 < output_size <= 84:
            raise ValueError("Output size must be between 1 and 84.")
        self.padding = nn.ZeroPad2d(2)
        self.conv_1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout2d()
        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout2d()
        self.flatten = nn.modules.Flatten()
        self.fc_1 = nn.Linear(16*5*5, 120)
        self.relu_3 = nn.ReLU()
        self.dropout_3 = nn.Dropout()
        self.fc_2 = nn.Linear(120, 84)
        self.relu_4 = nn.ReLU()
        self.dropout_4 = nn.Dropout()
        self.fc_3 = nn.Linear(84, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    @staticmethod  
    def create_hyper_parameters():
        return \
        {
            'dropout_rate_1': ContiniousHyperparameter(0.0, 1.0),
            'dropout_rate_2': ContiniousHyperparameter(0.0, 1.0),
            'dropout_rate_3': ContiniousHyperparameter(0.0, 1.0),
            'dropout_rate_4': ContiniousHyperparameter(0.0, 1.0)
        }

    def apply_hyper_parameters(self, hyper_parameters, device):
        self.dropout_1.p = hyper_parameters['dropout_rate_1'].value
        self.dropout_2.p = hyper_parameters['dropout_rate_2'].value
        self.dropout_3.p = hyper_parameters['dropout_rate_3'].value
        self.dropout_4.p = hyper_parameters['dropout_rate_4'].value