import random
import time
from collections import defaultdict

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import pbt.utils.data
from pbt.models.lenet5 import LeNet5
from pbt.database import ReadOnlyDatabase
from pbt.analyze import Analyzer 
from pbt.loss import CategoricalCrossEntropy, Accuracy, F1

class ControlTest():
    def test_lenet5_mnist(self):
        # this gets test_cce 0.03494, test_acc 99.07444 after 12 epochs
        epochs = 12
        batch_size = 64
        device = "cuda"
        model= LeNet5(10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
        loss_metric = 'cce'
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
                torchvision.transforms.Pad(padding=2, padding_mode='edge'),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_set = torchvision.datasets.MNIST(
            test_data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Pad(padding=2, padding_mode='edge'),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]))
        # split training set into training set and validation set
        train_set, eval_set = pbt.utils.data.stratified_split(
            train_data, labels=train_data.targets, fraction=50000/60000, random_state=1)

        train_data = torch.utils.data.DataLoader(
            dataset = train_set,
            batch_size = batch_size,
            shuffle = False)

        eval_data = torch.utils.data.DataLoader(
            dataset = eval_set,
            batch_size = batch_size,
            shuffle = False)

        test_data = torch.utils.data.DataLoader(
            dataset = test_set,
            batch_size = batch_size,
            shuffle = False)

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
            train_time = time.time_ns()
            train_loss = train(train_data, model, optimizer, loss_functions, device)
            train_time = float(time.time_ns() - train_time) * float(10**(-9))
            eval_time = time.time_ns()
            eval_loss = eval(eval_data, model, loss_functions, device)
            eval_time = float(time.time_ns() - eval_time) * float(10**(-9))
            for loss_name, loss_value in train_loss.items():
                result.append(f"train_{loss_name} {loss_value:.5f}")
            for loss_name, loss_value in eval_loss.items():
                result.append(f"eval_{loss_name} {loss_value:.5f}")
            print(f"Time: {train_time:.2f}s train, {eval_time:.2f}s eval")
            print(", ".join(result))
            train_scores.append(train_loss)
            eval_scores.append(eval_loss) 

        # test
        print("testing...")
        test_loss = test(test_data, model, loss_functions, device)
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