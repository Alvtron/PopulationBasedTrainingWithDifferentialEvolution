import random
from collections import defaultdict

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import context
import pbt.models
import pbt.utils.data
from pbt.database import ReadOnlyDatabase
from pbt.analyze import Analyzer 
from pbt.loss import CategoricalCrossEntropy, Accuracy, F1

# various settings for reproducibility
# set random state 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# set torch settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

def train(train_data, model, optimizer, loss_functions, device):
    train_loss = defaultdict(float)
    model.train()
    for x, y in train_data:
        x, y = x.to(device), y.to(device)
        for metric_type, metric_function in loss_functions.items():
            if metric_type == loss_metric:
                output = model(x)
                model.zero_grad()
                loss = metric_function(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss[metric_type] += loss.item() / len(train_data)
            else:
                loss = metric_function(output, y)
                train_loss[metric_type] += loss.item() / len(train_data)
    return train_loss

def eval(eval_data, model, loss_functions, device):
    eval_loss = defaultdict(float)
    model.eval()
    for x, y in eval_data:
        x, y = x.to(device), y.to(device)
        output = model(x)
        for metric_type, metric_function in loss_functions.items():
            loss = metric_function(output, y)
            eval_loss[metric_type] += loss.item() / len(eval_data)
    return eval_loss

def test(test_data, model, loss_functions, device):
    test_loss = defaultdict(float)
    model.eval()
    for x, y in test_data:
        x, y = x.to(device), y.to(device)
        output = model(x)
        for metric_type, metric_function in loss_functions.items():
            loss = metric_function(output, y)
            test_loss[metric_type] += loss.item() / len(eval_data)
    return test_loss

epochs = 100
batch_size = 64
device = "cuda"
model= pbt.models.MnistNet10Larger().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
loss_metric = 'cce'
eval_metric = 'acc'
loss_functions = {
    'cce': CategoricalCrossEntropy(),
    'acc': Accuracy()
}
# prepare training and testing data
train_data_path = test_data_path = './data'
train_data = torchvision.datasets.MNIST(
    train_data_path,
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]))
test_set = torchvision.datasets.MNIST(
    test_data_path,
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]))
# split training set into training set and validation set
train_set, _, eval_set, _ = pbt.utils.data.stratified_split(
    train_data, labels=train_data.targets, fraction=50000/60000, random_state=1)

train_data = torch.utils.data.DataLoader(
    dataset = train_set,
    batch_size = batch_size,
    shuffle = False,
    num_workers=0)

eval_data = torch.utils.data.DataLoader(
    dataset = eval_set,
    batch_size = batch_size,
    shuffle = False,
    num_workers=0)

test_data = torch.utils.data.DataLoader(
    dataset = test_set,
    batch_size = batch_size,
    shuffle = False,
    num_workers=0)

print(f"train samples: {len(train_set)}")
print(f"eval samples: {len(eval_set)}")
print(f"train batches: {len(train_data)}")
print(f"eval batches: {len(eval_data)}")

train_scores = list()
eval_scores = list()

# train and validate for e epochs
print("training...")
for e in range(epochs):
    result = list()
    result.append(f"epoch {e}")
    train_loss = train(train_data, model, optimizer, loss_functions, device)
    eval_loss = eval(eval_data, model, loss_functions, device)
    for loss_name, loss_value in train_loss.items():
        result.append(f"train_{loss_name} {loss_value:.5f}")
    for loss_name, loss_value in eval_loss.items():
        result.append(f"eval_{loss_name} {loss_value:.5f}")
    print(", ".join(result))
    train_scores.append(train_loss)
    eval_scores.append(eval_loss) 

# test
print("testing...")
test_loss = eval(test_data, model, loss_functions, device)
result = list()
for loss_name, loss_value in test_loss.items():
    result.append(f"test_{loss_name} {loss_value:.5f}")
print(", ".join(result))

# plotting
print("plotting...")
for metric_type in loss_functions:
    train_result_data = [loss_dict[metric_type] for loss_dict in train_scores]
    eval_result_data = [loss_dict[metric_type] for loss_dict in eval_scores]
    plt.plot(train_result_data, label=f"train_{metric_type}")
    plt.plot(eval_result_data, label=f"eval_{metric_type}")
    plt.legend()
    plt.show()