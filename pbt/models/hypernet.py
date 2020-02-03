import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod, abstractstaticmethod

class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()

    @abstractstaticmethod
    def create_hyper_parameters():
        pass

    @abstractmethod
    def apply_hyper_parameters(self, hyper_parameters, device):
        pass
    
    def forward(self, x):
        raise NotImplementedError()