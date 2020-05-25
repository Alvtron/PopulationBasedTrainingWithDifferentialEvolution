import time
import random
import itertools
import collections
from copy import deepcopy
from warnings import warn

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.vision import StandardTransform
from torch.optim import Optimizer

from .member import Checkpoint, MissingStateError
from .hyperparameters import Hyperparameters
from .models.hypernet import HyperNet
from pbt.utils.data import create_subset, create_subset_by_size

class Trainer(object):
    """ A class for training the provided model with the provided hyper-parameters on the set training dataset. """
    def __init__(self, model_class: HyperNet, optimizer_class: Optimizer, train_data: Dataset, batch_size: int,
            loss_functions: dict, loss_metric : str, step_size: int = 1, shuffle: bool = False, verbose: bool = False):
        if step_size < 1:
            raise ValueError("The number of steps must be at least one or higher.")
        self.LOSS_GROUP = 'train'
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.train_data = train_data
        self.step_size = step_size
        self.batch_size = batch_size
        self.loss_functions = loss_functions
        self.loss_metric = loss_metric
        self.shuffle = shuffle
        self.verbose = verbose

    def _print(self, message: str = None, end: str = '\n'):
        if not self.verbose:
            return
        print(message, end=end)

    def create_model(self, hyper_parameters: Hyperparameters = None, model_state: dict = None, device: str = 'cpu'):
        self._print("creating model...")
        model = self.model_class().to(device)
        if model_state is not None:
            self._print("loading model state...")
            model.load_state_dict(model_state)
        if isinstance(model, HyperNet) and hyper_parameters is not None and hasattr(hyper_parameters, 'model'):
            self._print("applying hyper-parameters...")
            model.apply_hyper_parameters(hyper_parameters.model, device)
        model.train()
        return model

    def create_optimizer(self, model: HyperNet, hyper_parameters: Hyperparameters, optimizer_state: dict  = None):
        self._print("creating optimizer...")
        get_value_dict = {hp_name:hp_value.value for hp_name, hp_value in hyper_parameters.optimizer.items()}
        optimizer = self.optimizer_class(model.parameters(), **get_value_dict)
        if optimizer_state:
            self._print("loading optimizer state...")
            optimizer.load_state_dict(optimizer_state)
            self._print("applying hyper-parameters...")
            for param_name, param_value in hyper_parameters.optimizer.items():
                for param_group in optimizer.param_groups:
                    param_group[param_name] = param_value.value
        return optimizer        

    def __call__(self, checkpoint: Checkpoint, device: str = 'cpu'):
        start_train_time_ns = time.time_ns()
        # load checkpoint state
        self._print(f"loading state of checkpoint {checkpoint.id}...")
        try:
            checkpoint.load_state(device=device, missing_ok=checkpoint.steps == 0)
        except MissingStateError:
            warn(f"checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.")
        # preparing model and optimizer
        self._print("creating model...")
        model = self.create_model(checkpoint.parameters, checkpoint.model_state, device)
        self._print("creating optimizer...")
        optimizer = self.create_optimizer(model, checkpoint.parameters, checkpoint.optimizer_state)
        # prepare batches
        self._print("creating batches...")
        subset = create_subset(dataset=self.train_data, start=checkpoint.steps * self.batch_size, end=(checkpoint.steps + self.step_size) * self.batch_size, shuffle=self.shuffle)
        batches = DataLoader(dataset = subset, batch_size = self.batch_size, shuffle = False)
        num_batches = len(batches)
        # reset loss dict
        checkpoint.loss[self.LOSS_GROUP] = dict.fromkeys(self.loss_functions, 0.0)
        self._print("Training...")
        for batch_index, (x, y) in enumerate(batches, 1):
            self._print(f"({batch_index}/{num_batches})", end=" ")
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # 1. Forward pass: compute predicted y by passing x to the model.
            output = model(x)
            for metric_type, metric_function in self.loss_functions.items():
                if metric_type == self.loss_metric:
                    # 2. Compute loss and save loss.
                    loss = metric_function(output, y)
                    checkpoint.loss[self.LOSS_GROUP][metric_type] += loss.item() / float(num_batches)
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
                    checkpoint.loss[self.LOSS_GROUP][metric_type] += loss.item() / float(num_batches)
                self._print(f"{metric_type}: {loss.item():4f}", end=" ")
                del loss
            del output
            checkpoint.steps += 1
            self._print(end="\n")
        # set new state
        checkpoint.model_state = model.state_dict()
        checkpoint.optimizer_state = optimizer.state_dict()
        # clean GPU memory
        del model
        del optimizer
        torch.cuda.empty_cache()
        # unload checkpoint state
        self._print(f"unloading state of checkpoint {checkpoint.id}...")
        checkpoint.unload_state()
        # save time
        checkpoint.time[self.LOSS_GROUP] = float(time.time_ns() - start_train_time_ns) * float(10**(-9))

class Evaluator(object):
    """ Class for evaluating the performance of the provided model on the set evaluation dataset. """
    def __init__(self, model_class: HyperNet, test_data: Dataset, batch_size: int, loss_functions: dict, loss_group: str = 'eval',
            batches: int = None, shuffle: bool = False, verbose: bool = False):
        self.model_class = model_class
        if batches is not None and batches < 1:
            raise ValueError("The number of batches must be at least one or higher.")
        if batches is not None:
            self.test_data = create_subset_by_size(dataset=test_data, n_samples = batches * batch_size, shuffle = shuffle)
            print(min(self.test_data.indices), "-->", max(self.test_data.indices), ", n =", len(self.test_data.indices), ", distinct:", len(self.test_data.indices)==len(set(self.test_data.indices)))
        else:
            self.test_data = test_data
        self.batch_size = batch_size
        self.loss_functions = loss_functions
        self.loss_group = loss_group
        self.shuffle = shuffle
        self.verbose = verbose

    def _print(self, message: str = None, end: str = '\n'):
        if not self.verbose:
            return
        print(message, end=end)

    def create_model(self, model_state: dict, device: str):
        self._print("creating model...")
        model = self.model_class().to(device)
        if model_state:
            model.load_state_dict(model_state)
        model.eval()
        return model

    def __call__(self, checkpoint: dict, device: str):
        """Evaluate model on the provided validation or test set."""
        start_eval_time_ns = time.time_ns()
        # load checkpoint state
        self._print(f"loading state of checkpoint {checkpoint.id}...")
        checkpoint.load_state(device=device, missing_ok=False)
        # preparing model
        self._print("creating model...")
        model = self.create_model(checkpoint.model_state, device)
        # prepare batches
        self._print("creating batches...")
        batches = DataLoader(dataset = self.test_data, batch_size = self.batch_size, shuffle = False)
        num_batches = len(batches)
        self._print("evaluating...")
        # reset loss dict
        checkpoint.loss[self.loss_group] = dict.fromkeys(self.loss_functions, 0.0)
        for batch_index, (x, y) in enumerate(batches, 1):
            if self.verbose: print(f"({batch_index}/{num_batches})", end=" ")
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                output = model(x)
            for metric_type, metric_function in self.loss_functions.items():
                with torch.no_grad():
                    loss = metric_function(output, y)
                checkpoint.loss[self.loss_group][metric_type] += loss.item() / float(num_batches)
                if self.verbose: print(f"{metric_type}: {loss.item():4f}", end=" ")
                del loss
            if self.verbose: print(end="\n")
            del output
        # clean GPU memory
        del model
        torch.cuda.empty_cache()
        # unload checkpoint state
        self._print(f"unloading state of checkpoint {checkpoint.id}...")
        checkpoint.unload_state()
        # save time
        checkpoint.time[self.loss_group] = float(time.time_ns() - start_eval_time_ns) * float(10**(-9))

class RandomFitnessApproximation():
    def __init__(self, model_class: HyperNet, optimizer_class: Optimizer, train_data: Dataset, test_data: Dataset, batches: int, batch_size : int,
            loss_functions: dict, loss_metric: str, verbose: bool = False):
        # n batches for training
        self.trainer = Trainer(model_class=model_class, optimizer_class=optimizer_class, train_data=train_data, step_size=batches,
            batch_size=batch_size, loss_functions=loss_functions, loss_metric=loss_metric, shuffle=False, verbose=verbose)
        # n random batches for evaluation
        self.evaluator = Evaluator(model_class=model_class, test_data=test_data, batches=batches,
            batch_size=batch_size, loss_functions=loss_functions, loss_group='eval', shuffle=True, verbose=verbose)
        self.weight = (batches * batch_size) / len(test_data)

    def __adjust_loss(self, previous_loss: dict, fitness_loss: dict) -> dict:
        new_loss = collections.defaultdict(dict)
        for loss_group in fitness_loss:
            for loss_type in fitness_loss[loss_group]:
                previous_loss_value = previous_loss[loss_group][loss_type]
                fitness_loss_value = fitness_loss[loss_group][loss_type]
                new_loss[loss_group][loss_type] = (previous_loss_value * (1 - self.weight)) + (fitness_loss_value * self.weight)
        return new_loss

    def __call__(self, checkpoint: Checkpoint, device: str):
        old_loss = deepcopy(checkpoint.loss)
        self.trainer(checkpoint, device)
        self.evaluator(checkpoint, device)
        checkpoint.loss = self.__adjust_loss(old_loss, checkpoint.loss)