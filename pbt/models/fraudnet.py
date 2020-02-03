import math
from abc import abstractmethod, abstractstaticmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypernet import HyperNet
from ..hyperparameters import Hyperparameter

class FraudNet(HyperNet):
    def __init__(self):
        super(FraudNet, self).__init__()
        self.fc_1 = nn.Linear(30, 16)
        self.prelu_1 = nn.PReLU()
        self.fc_2 = nn.Linear(16, 18)
        self.prelu_2 = nn.PReLU()
        self.dropout_1 = nn.Dropout()
        self.fc_3 = nn.Linear(18, 20)
        self.prelu_3 = nn.PReLU()
        self.fc_4 = nn.Linear(20, 24)
        self.prelu_4 = nn.PReLU()
        self.fc_5 = nn.Linear(24, 1)
        self.sigmoid = nn.Sigmoid()
        
    @staticmethod
    def create_hyper_parameters():
        return \
        { 
            'dropout_rate_1': Hyperparameter(0.0, 1.0),
            'prelu_alpha_1': Hyperparameter(0.0, 1.0),
            'prelu_alpha_2': Hyperparameter(0.0, 1.0),
            'prelu_alpha_3': Hyperparameter(0.0, 1.0),
            'prelu_alpha_4': Hyperparameter(0.0, 1.0)
        }

    def apply_hyper_parameters(self, hyper_parameters, device):
        self.dropout_1.p = hyper_parameters['dropout_rate_1'].value
        self.prelu_1 = nn.PReLU(init=hyper_parameters['prelu_alpha_1'].value).to(device)
        self.prelu_2 = nn.PReLU(init=hyper_parameters['prelu_alpha_2'].value).to(device)
        self.prelu_3 = nn.PReLU(init=hyper_parameters['prelu_alpha_3'].value).to(device)
        self.prelu_4 = nn.PReLU(init=hyper_parameters['prelu_alpha_4'].value).to(device)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x