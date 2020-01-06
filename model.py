import torch.nn as nn

class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()

    def apply_hyper_parameters(self, hyper_parameters, device):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

class MnistNet(HyperNet):
    def __init__(self):
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
        self.fc_2 = nn.Linear(50, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def apply_hyper_parameters(self, hyper_parameters, device):
        self.dropout_1.p = hyper_parameters['dropout_rate_1'].value()
        self.dropout_2.p = hyper_parameters['dropout_rate_2'].value()
        self.prelu_1 = nn.PReLU(init=hyper_parameters['prelu_alpha_1'].value()).to(device)
        self.prelu_2 = nn.PReLU(init=hyper_parameters['prelu_alpha_2'].value()).to(device)
        self.prelu_3 = nn.PReLU(init=hyper_parameters['prelu_alpha_3'].value()).to(device)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.max_pool_1(x)
        x = self.prelu_1(x)
        x = self.conv_2(x)
        x = self.dropout_1(x)
        x = self.max_pool_2(x)
        x = self.prelu_2(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.prelu_3(x)
        x = self.dropout_2(x)
        x = self.fc_2(x)
        x = self.softmax(x)
        return x
        

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

    def apply_hyper_parameters(self, hyper_parameters, device):
        self.dropout_1.p = hyper_parameters['dropout_rate_1'].value()
        self.prelu_1 = nn.PReLU(init=hyper_parameters['prelu_alpha_1'].value()).to(device)
        self.prelu_2 = nn.PReLU(init=hyper_parameters['prelu_alpha_2'].value()).to(device)
        self.prelu_3 = nn.PReLU(init=hyper_parameters['prelu_alpha_3'].value()).to(device)
        self.prelu_4 = nn.PReLU(init=hyper_parameters['prelu_alpha_4'].value()).to(device)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.prelu_1(x)
        x = self.fc_2(x)
        x = self.prelu_2(x)
        x = self.dropout_1(x)
        x = self.fc_3(x)
        x = self.prelu_3(x)
        x = self.fc_4(x)
        x = self.prelu_4(x)
        x = self.fc_5(x)
        x = self.sigmoid(x)
        return x