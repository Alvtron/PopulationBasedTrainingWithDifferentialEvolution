import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import Hyperparameter
from abc import abstractmethod, abstractstaticmethod

class BasicBlock(nn.Module):
    """https://github.com/xternalz/WideResNet-pytorch"""
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    """https://github.com/xternalz/WideResNet-pytorch"""
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    """https://github.com/xternalz/WideResNet-pytorch"""
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.modules.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_1 = nn.Conv2d(1, 6, kernel_size=(5,5))
        self.relu_1 = nn.ReLU()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv_2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.relu_2 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.flatten = nn.modules.Flatten()
        self.fc_1 = nn.Linear(120, 84)
        self.relu_4 = nn.ReLU()
        self.fc_2 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

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
            'dropout_rate_1': Hyperparameter(0.0, 1.0),
            'dropout_rate_2': Hyperparameter(0.0, 1.0),
            'prelu_alpha_1': Hyperparameter(0.0, 1.0),
            'prelu_alpha_2': Hyperparameter(0.0, 1.0),
            'prelu_alpha_3': Hyperparameter(0.0, 1.0),
        }

    def apply_hyper_parameters(self, hyper_parameters, device):
        self.dropout_1.p = hyper_parameters['dropout_rate_1'].value
        self.dropout_2.p = hyper_parameters['dropout_rate_2'].value
        self.prelu_1 = nn.PReLU(init=hyper_parameters['prelu_alpha_1'].value).to(device)
        self.prelu_2 = nn.PReLU(init=hyper_parameters['prelu_alpha_2'].value).to(device)
        self.prelu_3 = nn.PReLU(init=hyper_parameters['prelu_alpha_3'].value).to(device)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class MnistNet10(MnistNet):
    def __init__(self):
        super().__init__(output_size=10)

class MnistNet26(MnistNet):
    def __init__(self):
        super().__init__(output_size=26)

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
    def create_hyper_parameters():
        return \
        {
            'dropout_rate_1': Hyperparameter(0.0, 1.0),
            'dropout_rate_2': Hyperparameter(0.0, 1.0),
            'dropout_rate_3': Hyperparameter(0.0, 1.0),
            'prelu_alpha_1': Hyperparameter(0.0, 1.0),
            'prelu_alpha_2': Hyperparameter(0.0, 1.0),
            'prelu_alpha_3': Hyperparameter(0.0, 1.0),
            'prelu_alpha_4': Hyperparameter(0.0, 1.0),
            'prelu_alpha_5': Hyperparameter(0.0, 1.0),
            'prelu_alpha_6': Hyperparameter(0.0, 1.0)
        }

    def apply_hyper_parameters(self, hyper_parameters, device):
        self.dropout_1.p = hyper_parameters['dropout_rate_1'].value
        self.dropout_2.p = hyper_parameters['dropout_rate_2'].value
        self.dropout_3.p = hyper_parameters['dropout_rate_3'].value
        self.prelu_1 = nn.PReLU(init=hyper_parameters['prelu_alpha_1'].value).to(device)
        self.prelu_2 = nn.PReLU(init=hyper_parameters['prelu_alpha_2'].value).to(device)
        self.prelu_3 = nn.PReLU(init=hyper_parameters['prelu_alpha_3'].value).to(device)
        self.prelu_4 = nn.PReLU(init=hyper_parameters['prelu_alpha_4'].value).to(device)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class MnistNet10Larger(MnistNetLarger):
    def __init__(self):
        super().__init__(output_size=10)

class MnistNet26Larger(MnistNetLarger):
    def __init__(self):
        super().__init__(output_size=26)

class MnistNet47Larger(MnistNetLarger):
    def __init__(self):
        super().__init__(output_size=47)

class MnistNet62Larger(MnistNetLarger):
    def __init__(self):
        super().__init__(output_size=62)

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
            'dropout_rate_1': Hyperparameter(0.0, 1.0),
            'dropout_rate_2': Hyperparameter(0.0, 1.0),
            'dropout_rate_3': Hyperparameter(0.0, 1.0),
            'prelu_alpha_1': Hyperparameter(0.0, 1.0),
            'prelu_alpha_2': Hyperparameter(0.0, 1.0),
            'prelu_alpha_3': Hyperparameter(0.0, 1.0),
            'prelu_alpha_4': Hyperparameter(0.0, 1.0),
            'prelu_alpha_5': Hyperparameter(0.0, 1.0),
            'prelu_alpha_6': Hyperparameter(0.0, 1.0)
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

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class MnistNet10Largest(MnistNetLargest):
    def __init__(self):
        super().__init__(output_size=10)

class MnistNet26Largest(MnistNetLargest):
    def __init__(self):
        super().__init__(output_size=26)

class MnistNet47Largest(MnistNetLargest):
    def __init__(self):
        super().__init__(output_size=47)

class MnistNet62Largest(MnistNetLargest):
    def __init__(self):
        super().__init__(output_size=62)