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

# various settings for reproducibility
# set random state 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# set torch settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

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
    loss_functions=task.loss_functions)
tester = Evaluator(
    model_class=task.model_class,
    test_data=task.datasets.test,
    batch_size=batch_size,
    loss_functions=task.loss_functions)
hp = Hyperparameters(
    augment_params=None,
    model_params=None,
    optimizer_params={
        'lr': ContiniousHyperparameter(0.000001, 0.1, value=0.01),
        'momentum': ContiniousHyperparameter(0.001, 0.9, value=0.5)
    })

step_size = 100
checkpoint = Checkpoint(
    id=0,
    hyper_parameters=hp,
    loss_metric=loss_metric,
    eval_metric=eval_metric,
    minimize=loss_functions[eval_metric].minimize)

# train and validate for e epochs
print("training...")
while(checkpoint.epochs < epochs):
    start_train_time_ns = time.time_ns()
    checkpoint.model_state, checkpoint.optimizer_state, checkpoint.epochs, checkpoint.steps, checkpoint.loss['train'] = trainer(
        hyper_parameters=checkpoint.hyper_parameters,
        model_state=checkpoint.model_state,
        optimizer_state=checkpoint.optimizer_state,
        epochs=checkpoint.epochs,
        steps=checkpoint.steps,
        step_size=step_size,
        device=device)
    checkpoint.time['train'] = float(time.time_ns() - start_train_time_ns) * float(10**(-9))
    start_eval_time_ns = time.time_ns()
    checkpoint.loss['eval'] = evaluator(checkpoint.model_state, device)
    checkpoint.time['eval'] = float(time.time_ns() - start_eval_time_ns) * float(10**(-9))
    print(f"Time: {checkpoint.time['train']:.2f}s train, {checkpoint.time['eval']:.2f}s eval")
    print(f"epoch {checkpoint.epochs}, step {checkpoint.steps}, {checkpoint.performance_details()}")
    checkpoint.hyper_parameters.optimizer['lr'] *= random.uniform(0.8, 1.2)
    checkpoint.hyper_parameters.optimizer['momentum'] *= random.uniform(0.8, 1.2)
# test
print("testing...")
checkpoint.loss['test'] = tester(checkpoint.model_state, device)
print(checkpoint.performance_details())