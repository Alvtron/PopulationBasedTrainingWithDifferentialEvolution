import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod, abstractstaticmethod

class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()

    @staticmethod 
    def create_hyper_parameters():
        return None

    def apply_hyper_parameters(self, hyper_parameters, device):
        pass
    
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x