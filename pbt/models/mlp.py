import math
from abc import abstractmethod, abstractstaticmethod

import torch.nn as nn

from .hypernet import HyperNet
from ..hyperparameters import ContiniousHyperparameter

class MLP(HyperNet):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.modules.Flatten()
        self.fc1 = nn.Linear(1024, 128)
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(dim=1)