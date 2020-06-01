import math
from abc import abstractmethod, abstractstaticmethod

import torch.nn as nn

from .hypernet import HyperNet

class MLP(HyperNet):
    def __init__(self, output_size: int):
        super(MLP, self).__init__()
        if not 0 < output_size <= 64:
            raise ValueError("Output size must be between 1 and 64.")
        self.flatten = nn.modules.Flatten()
        self.fc1 = nn.Linear(1024, 128)
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.LogSoftmax(dim=1)