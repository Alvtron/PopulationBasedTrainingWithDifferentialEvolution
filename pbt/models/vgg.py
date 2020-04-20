import torch
import torch.nn as nn
from torchvision.models import vgg16

from .hypernet import HyperNet

class VGG16(HyperNet):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = vgg16(pretrained=False)