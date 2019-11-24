import torch.nn as nn
import torch.nn.functional as F

class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.conv1_pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_pool = nn.MaxPool2d(kernel_size=2)
        self.conv2_dropout = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc1_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def apply_hyper_parameters(self, hyper_parameters):
        self.conv2_dropout.p = hyper_parameters['dropout_rate_1'].value()
        self.fc1_dropout.p = hyper_parameters['dropout_rate_2'].value()

    def forward(self, x):
        x = self.relu1(self.conv1_pool(self.conv1(x)))
        x = self.relu2(self.conv2_pool(self.conv2_dropout(self.conv2(x))))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
