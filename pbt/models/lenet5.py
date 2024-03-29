import torch.nn as nn

from pbt.models.hypernet import HyperNet
from pbt.hyperparameters import ContiniousHyperparameter

class LeNet5(HyperNet):
    def __init__(self, output_size: int):
        super().__init__()
        if not 0 < output_size <= 84:
            raise ValueError("Output size must be between 1 and 84.")
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_2 = nn.ReLU()
        self.flatten = nn.modules.Flatten()
        self.fc_1 = nn.Linear(5*5*16, 120)
        self.relu_3 = nn.ReLU()
        self.fc_2 = nn.Linear(120, 84)
        self.relu_4 = nn.ReLU()
        self.fc_3 = nn.Linear(84, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

class Lenet5WithDropout(HyperNet):
    def __init__(self, output_size: int):
        super().__init__()
        if not 0 < output_size <= 84:
            raise ValueError("Output size must be between 1 and 84.")
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout2d()
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout2d()
        self.flatten = nn.modules.Flatten()
        self.fc_1 = nn.Linear(5*5*16, 120)
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

class Lenet5WithBatchNorm(HyperNet):
    def __init__(self, output_size: int):
        super().__init__()
        if not 0 < output_size <= 84:
            raise ValueError("Output size must be between 1 and 84.")
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout2d()
        self.batch_norm_1 = nn.BatchNorm2d(6)
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout2d()
        self.batch_norm_2 = nn.BatchNorm2d(16)
        self.flatten = nn.modules.Flatten()
        self.fc_1 = nn.Linear(5*5*16, 120)
        self.relu_3 = nn.ReLU()
        self.dropout_3 = nn.Dropout()
        self.batch_norm_3 = nn.BatchNorm1d(120)
        self.fc_2 = nn.Linear(120, 84)
        self.relu_4 = nn.ReLU()
        self.dropout_4 = nn.Dropout()
        self.batch_norm_4 = nn.BatchNorm1d(84)
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