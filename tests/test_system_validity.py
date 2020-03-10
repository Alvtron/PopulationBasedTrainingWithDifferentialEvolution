import random
import time
from collections import defaultdict
from functools import partial

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import context
import pbt.utils.data
from pbt.models.lenet5 import LeNet5
from pbt.database import ReadOnlyDatabase
from pbt.analyze import Analyzer 
from pbt.loss import CategoricalCrossEntropy, Accuracy, F1
from pbt.trainer import Trainer
from pbt.evaluator import Evaluator
from pbt.hyperparameters import Hyperparameters, ContiniousHyperparameter, DiscreteHyperparameter
from pbt.member import Checkpoint
from pbt.task.mnist import MnistKnowledgeSharing

epochs = 100
batch_size = 64
device = "cuda"
model = LeNet5
optimizer = torch.optim.SGD
loss_metric = 'cce'
eval_metric = 'acc'
loss_functions = {
    'cce': CategoricalCrossEntropy(),
    'acc': Accuracy()
}

task = MnistKnowledgeSharing('lenet5')

trainer = Trainer(
    model_class=task.model_class,
    optimizer_class=task.optimizer_class,
    train_data=task.datasets.train,
    batch_size=batch_size,
    loss_functions=task.loss_functions,
    loss_metric=task.loss_metric)
evaluator = Evaluator(
    model_class=task.model_class,
    test_data=task.datasets.eval,
    batch_size=batch_size,
    loss_group='eval',
    loss_functions=task.loss_functions)
tester = Evaluator(
    model_class=task.model_class,
    test_data=task.datasets.test,
    batch_size=batch_size,
    loss_group='test',
    loss_functions=task.loss_functions)
hp = Hyperparameters(
    optimizer={
        'lr': ContiniousHyperparameter(0.000001, 0.1, value=0.01),
        'momentum': ContiniousHyperparameter(0.001, 0.9, value=0.5)
    })

step_size = 100
checkpoint = Checkpoint(
    id=0,
    parameters=hp,
    loss_metric=loss_metric,
    eval_metric=eval_metric,
    minimize=loss_functions[eval_metric].minimize)

# train and validate for e epochs
print("training...")
while(checkpoint.epochs < epochs):
    start_train_time_ns = time.time_ns()
    trainer(checkpoint, step_size, device)
    checkpoint.time['train'] = float(time.time_ns() - start_train_time_ns) * float(10**(-9))
    start_eval_time_ns = time.time_ns()
    evaluator(checkpoint, device)
    checkpoint.time['eval'] = float(time.time_ns() - start_eval_time_ns) * float(10**(-9))
    print(f"Time: {checkpoint.time['train']:.2f}s train, {checkpoint.time['eval']:.2f}s eval")
    print(f"epoch {checkpoint.epochs}, step {checkpoint.steps}, {checkpoint.performance_details()}")
    checkpoint.parameters.optimizer['lr'] *= random.uniform(0.8, 1.2)
    checkpoint.parameters.optimizer['momentum'] *= random.uniform(0.8, 1.2)
# test
print("testing...")
tester(checkpoint, device)
print(checkpoint.performance_details())