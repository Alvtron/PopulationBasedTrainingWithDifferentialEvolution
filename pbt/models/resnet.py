import torch
import torch.nn as nn
import torchvision.models.resnet

from .hypernet import HyperNet

class ResNet18(torchvision.models.resnet.ResNet):
    def __init__(self, num_classes, num_input_channels):
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)