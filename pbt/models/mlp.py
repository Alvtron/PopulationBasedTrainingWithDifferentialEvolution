import math
from abc import abstractmethod, abstractstaticmethod

import torch.nn as nn

from .hypernet import HyperNet
from ..hyperparameters import Hyperparameter

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.modules.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)