import os
import random
import collections
from abc import ABC
from copy import deepcopy
from typing import Callable
from functools import partial

import torch
import torchvision
from torch.nn import Module
from torch.utils.data import Dataset, Subset, DataLoader
from torch.optim import Optimizer

from pbt.member import Checkpoint
from pbt.hyperparameters import Hyperparameters
from pbt.device import get_global_device
from pbt.models.hypernet import HyperNet
from pbt.utils.data import create_subset, create_subset_by_size
from pbt.nn import Trainer, Evaluator

def adjust_weighted_loss(weight: float, previous_loss: dict, fitness_loss: dict) -> dict:
    assert isinstance(previous_loss, dict), "previous_loss is wrong type"
    assert isinstance(fitness_loss, dict), "fitness_loss is wrong type"
    new_loss = collections.defaultdict(dict)
    for loss_group in fitness_loss:
        for loss_type in fitness_loss[loss_group]:
            fitness_loss_value = fitness_loss[loss_group][loss_type]
            # if there is no previous loss value, use the fitness value
            if loss_group not in previous_loss or loss_type not in previous_loss[loss_group]:
                new_loss[loss_group][loss_type] = fitness_loss_value
            previous_loss_value = previous_loss[loss_group][loss_type]
            new_loss[loss_group][loss_type] = (previous_loss_value * (1 - weight)) + (fitness_loss_value * weight)
    return new_loss

def rfa(checkpoint: Checkpoint, trainer: Trainer, evaluator: Evaluator, weight: float, device: str = None) -> None:
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError(f"the 'checkpoint' specified was of wrong type {type(checkpoint)}, expected {Checkpoint}.")
    if device is None:
        device = get_global_device()
    if not isinstance(device, str):
        raise TypeError(f"the 'device' specified was of wrong type {type(device)}, expected {str}.")
    # copy old loss
    old_loss = deepcopy(checkpoint.loss)
    # load checkpoint state
    checkpoint.load_state(device=device, missing_ok=True)
    # train and evaluate
    trainer(checkpoint, device)
    evaluator(checkpoint, device)
    # unload checkpoint state
    checkpoint.unload_state()
    # correct loss
    checkpoint.loss = adjust_weighted_loss(weight, old_loss, checkpoint.loss)

class FitnessFunctionProvider(ABC):
    def __enter__(self) -> Callable[[Checkpoint], None]:
        raise NotImplementedError()

    def __exit__(self, exception_type, exception_value, traceback):
        raise NotImplementedError()

class RandomFitnessApproximation(FitnessFunctionProvider):
    def __init__(self, model_class: HyperNet, optimizer_class: Optimizer, train_data: Dataset, test_data: Dataset, batches: int, batch_size: int,
                 loss_functions: dict, loss_metric: str, verbose: bool = False):
        # n batches for training
        self.__trainer = Trainer(
            model_class=model_class, optimizer_class=optimizer_class, train_data=train_data, step_size=batches,
            batch_size=batch_size, loss_functions=loss_functions, loss_metric=loss_metric)
        # n random batches for evaluation
        self.__partial_evaluator = partial(Evaluator, 
            model_class=model_class, test_data=test_data, batches=batches,
            batch_size=batch_size, loss_functions=loss_functions, loss_group='eval',
            shuffle=True)
        self.weight = (batches * batch_size) / len(test_data)

    def __enter__(self) -> Callable[[Checkpoint], None]:
        return partial(rfa, trainer=self.__trainer, evaluator=self.__partial_evaluator(), weight=self.weight)

    def __exit__(self, exception_type, exception_value, traceback):
        pass