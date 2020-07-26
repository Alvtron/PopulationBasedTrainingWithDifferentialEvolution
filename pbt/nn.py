import os
import random
import itertools
import collections
from abc import ABC
from copy import deepcopy
from warnings import warn
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

class Trainer(object):
    """ A class for training the provided model with the provided hyper-parameters on the set training dataset. """

    def __init__(self, model_class, optimizer_class, train_data: Dataset, batch_size: int, loss_functions: dict, loss_metric: str, step_size: int = 1):
        if not callable(model_class):
            raise TypeError(f"the 'model_class' specified was not callable.")
        if not callable(optimizer_class):
            raise TypeError(f"the 'optimizer_class' specified was not callable.")
        if not isinstance(train_data, Dataset):
            raise TypeError(f"the 'train_data' specified was of wrong type {type(train_data)}, expected {Dataset}.")
        if not isinstance(batch_size, int):
            raise TypeError(f"the 'batch_size' specified was of wrong type {type(batch_size)}, expected {int}.")
        if not isinstance(loss_functions, dict):
            raise TypeError(f"the 'loss_functions' specified was of wrong type {type(loss_functions)}, expected {dict}.")
        if not isinstance(loss_metric, str):
            raise TypeError(f"the 'loss_metric' specified was of wrong type {type(loss_metric)}, expected {str}.")
        if not isinstance(step_size, int):
            raise TypeError(f"the 'step_size' specified was of wrong type {type(step_size)}, expected {int}.")
        if step_size < 1:
            raise ValueError("The 'step_size' specified was lower than 1.")
        self.LOSS_GROUP = 'train'
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.train_data = train_data
        self.step_size = step_size
        self.batch_size = batch_size
        self.loss_functions = loss_functions
        self.loss_metric = loss_metric

    def __create_model(self, hyper_parameters: Hyperparameters, device: str, model_state: dict = None) -> Module:
        """Create an instance of the model with the supplied hyper-parameters and optional model state"""
        assert isinstance(hyper_parameters, Hyperparameters), "hyper_parameters is wrong type"
        assert isinstance(device, str), "device is of wrong type"
        assert model_state is None or isinstance(model_state, dict), "model_state is wrong type"
        # create model instance
        model = self.model_class().to(device)
        # loading model state
        if model_state is not None:
            model.load_state_dict(model_state)
        # applying hyper-parameters
        if isinstance(model, HyperNet) and hasattr(hyper_parameters, 'model'):
            model.apply_hyper_parameters(hyper_parameters.model, device)
        # set model to train mode
        model.train()
        return model

    def __create_optimizer(self, model: HyperNet, hyper_parameters: Hyperparameters, optimizer_state: dict = None) -> Optimizer:
        assert isinstance(model, Module), "model is of wrong type"
        assert isinstance(hyper_parameters, Hyperparameters), "hyper_parameters is wrong type"
        assert optimizer_state is None or isinstance(optimizer_state, dict), "optimizer_state is wrong type"
        hp_value_dict = {hp_name: hp_value.value for hp_name, hp_value in hyper_parameters.optimizer.items()}
        optimizer = self.optimizer_class(model.parameters(), **hp_value_dict)
        if optimizer_state is not None:
            # loading optimizer state
            optimizer.load_state_dict(optimizer_state)
            # applying hyper-parameters
            for param_name, param_value in hyper_parameters.optimizer.items():
                for param_group in optimizer.param_groups:
                    param_group[param_name] = param_value.value
        return optimizer

    def __create_dataloader(self, steps_performed : int) -> DataLoader:
        """Load a cyclic subset of the training data with indices decided by the range between the number of steps performed and step size, and return the dalaloader.\n
        If the number of steps exceeds the length of the training data, the exceeding steps are shifted to index 0.
        """
        assert isinstance(steps_performed, int), "start step is wrong type"
        assert steps_performed >= 0, "steps_performed was negative"
        dataset_size = len(self.train_data)
        n_samples = self.step_size * self.batch_size
        start_index = (steps_performed * self.batch_size) % dataset_size
        end_index = start_index + n_samples
        indices = list(itertools.islice(itertools.cycle(range(dataset_size)), start_index, end_index))
        subset = Subset(dataset=self.train_data, indices=indices)
        return DataLoader(dataset = subset, batch_size = self.batch_size, shuffle = False)

    def __calculate_epochs(self, steps_performed: int) -> int:
        """Calculate the number of epochs based on the number of steps performed."""
        samples_processed = (steps_performed * self.batch_size)
        return samples_processed // len(self.train_data)

    def __call__(self, checkpoint : Checkpoint, device : str = None):
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError(f"the 'checkpoint' specified was of wrong type {type(checkpoint)}, expected {Checkpoint}.")
        if device is None:
            device = get_global_device()
        if not isinstance(device, str):
            raise TypeError(f"the 'device' specified was of wrong type {type(device)}, expected {str}.")
        # preparing model and optimizer
        model = self.__create_model(hyper_parameters=checkpoint.parameters, model_state=checkpoint.model_state, device=device)
        optimizer = self.__create_optimizer(model=model, hyper_parameters=checkpoint.parameters, optimizer_state=checkpoint.optimizer_state)
        # create dataloader
        dataloader = self.__create_dataloader(steps_performed = checkpoint.steps)
        # reset loss dict
        checkpoint.loss[self.LOSS_GROUP] = dict.fromkeys(self.loss_functions, 0.0)
        # train
        for batch_index, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # 1. Forward pass: compute predicted y by passing x to the model.
            output = model(x)
            for metric_type, metric_function in self.loss_functions.items():
                if metric_type == self.loss_metric:
                    # 2. Compute loss and save loss.
                    loss = metric_function(output, y)
                    checkpoint.loss[self.LOSS_GROUP][metric_type] += loss.item() / float(self.step_size)
                    # 3. Before the backward pass, use the optimizer object to zero all of the gradients
                    # for the variables it will update (which are the learnable weights of the model).
                    optimizer.zero_grad()
                    # 4. Backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # 5. Calling the step function on an Optimizer makes an update to its parameters
                    optimizer.step()
                else:
                    # 2. Compute loss and save loss
                    with torch.no_grad():
                        loss = metric_function(output, y)
                    checkpoint.loss[self.LOSS_GROUP][metric_type] += loss.item() / float(self.step_size)
                del loss
            del output
            checkpoint.steps += 1
        # update number of epochs performed
        checkpoint.epochs = self.__calculate_epochs(checkpoint.steps)
        # set new state
        checkpoint.model_state = model.state_dict()
        checkpoint.optimizer_state = optimizer.state_dict()
        # clean GPU memory
        del model
        del optimizer
        torch.cuda.empty_cache()


class Evaluator(object):
    """ Class for evaluating the performance of the provided model on the set evaluation dataset. """

    def __init__(self, model_class, test_data: Dataset, batch_size: int, loss_functions: dict, loss_group: str = 'eval', batches: int = None, shuffle: bool = False):
        if not callable(model_class):
            raise TypeError(f"the 'model_class' specified was not callable.")
        if not isinstance(test_data, Dataset):
            raise TypeError(f"the 'test_data' specified was of wrong type {type(test_data)}, expected {Dataset}.")
        if not isinstance(batch_size, int):
            raise TypeError(f"the 'batch_size' specified was of wrong type {type(batch_size)}, expected {int}.")
        if not isinstance(loss_functions, dict):
            raise TypeError(f"the 'loss_functions' specified was of wrong type {type(loss_functions)}, expected {dict}.")
        if not isinstance(loss_group, str):
            raise TypeError(f"the 'loss_group' specified was of wrong type {type(loss_group)}, expected {str}.")
        if batches is not None:
            if not isinstance(batches, int):
                raise TypeError(f"the 'batches' specified was of wrong type {type(batches)}, expected {int}.")
            if batches < 1:
                raise ValueError("The 'batches' specified was less than 1.")
        if not isinstance(shuffle, bool):
            raise TypeError(f"the 'shuffle' specified was of wrong type {type(shuffle)}, expected {bool}.")
        self.model_class = model_class
        self.test_data = create_subset_by_size(
            dataset=test_data, n_samples=batches * batch_size, shuffle=shuffle) if batches is not None else test_data
        self.batch_size = batch_size
        self.loss_functions = loss_functions
        self.loss_group = loss_group
        self.shuffle = shuffle

    def __create_model(self, model_state: dict, device: str):
        assert isinstance(model_state, dict), "model_state is wrong type"
        assert isinstance(device, str), "device is of wrong type"
        model = self.model_class().to(device)
        model.load_state_dict(model_state)
        model.eval()
        return model

    def __call__(self, checkpoint: Checkpoint, device: str = None):
        """Evaluate checkpoint model."""
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError(f"the 'checkpoint' specified was of wrong type {type(checkpoint)}, expected {Checkpoint}.")
        if device is None:
            device = get_global_device()
        if not isinstance(device, str):
            raise TypeError(f"the 'device' specified was of wrong type {type(device)}, expected {str}.")
        # preparing model
        model = self.__create_model(model_state=checkpoint.model_state, device=device)
        # prepare batches
        batches = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        num_batches = len(batches)
        # reset loss dict
        checkpoint.loss[self.loss_group] = dict.fromkeys(self.loss_functions, 0.0)
        # evaluate
        for x, y in batches:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                output = model(x)
            for metric_type, metric_function in self.loss_functions.items():
                with torch.no_grad():
                    loss = metric_function(output, y)
                checkpoint.loss[self.loss_group][metric_type] += loss.item() / float(num_batches)
                del loss
            del output
        # clean GPU memory
        del model
        torch.cuda.empty_cache()


class Step():
    def __init__(self, model_class: HyperNet, optimizer_class: Optimizer, train_data: Dataset, test_data: Dataset, step_size: int, batch_size: int,
                 loss_functions: dict, loss_metric: str):
        # n batches for training
        self.trainer = Trainer(
            model_class=model_class, optimizer_class=optimizer_class, train_data=train_data, step_size=step_size,
            batch_size=batch_size, loss_functions=loss_functions, loss_metric=loss_metric)
        # n random batches for evaluation
        self.evaluator = Evaluator(
            model_class=model_class, test_data=test_data, batch_size=batch_size, loss_functions=loss_functions, loss_group='eval', shuffle=False)

    def __call__(self, checkpoint: Checkpoint, device: str = None):
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError(f"the 'checkpoint' specified was of wrong type {type(checkpoint)}, expected {Checkpoint}.")
        if device is None:
            device = get_global_device()
        if not isinstance(device, str):
            raise TypeError(f"the 'device' specified was of wrong type {type(device)}, expected {str}.")
        # load checkpoint state
        checkpoint.load_state(device=device, missing_ok=checkpoint.steps == 0)
        # train and evaluate
        self.trainer(checkpoint, device)
        self.evaluator(checkpoint, device)
        # unload checkpoint state
        checkpoint.unload_state()