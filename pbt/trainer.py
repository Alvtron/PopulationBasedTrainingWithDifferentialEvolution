import time
import itertools
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.vision import StandardTransform
from torch.optim import Optimizer

from .member import Checkpoint
from .hyperparameters import Hyperparameters
from .models.hypernet import HyperNet

class Trainer(object):
    """ A class for training the provided model with the provided hyper-parameters on the set training dataset. """
    def __init__(self, model_class : HyperNet, optimizer_class : Optimizer, train_data : Dataset, batch_size : int,
            loss_functions : dict, loss_metric : str, verbose : bool = False):
        self.LOSS_GROUP = 'train'
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.train_data = train_data
        self.batch_size = batch_size
        self.loss_functions = loss_functions
        self.loss_metric = loss_metric
        self.verbose = verbose

    def _print(self, message : str = None, end : str = '\n'):
        if not self.verbose:
            return
        print(message, end=end)

    def create_model(self, hyper_parameters : Hyperparameters = None, model_state = None, device: str = 'cpu'):
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

    def create_optimizer(self, model : HyperNet, hyper_parameters : Hyperparameters, optimizer_state : dict  = None):
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

    def __call__(self, checkpoint : Checkpoint, device: str = 'cpu'):
        start_train_time_ns = time.time_ns()
        # preparing model and optimizer
        self._print("creating model...")
        model = self.create_model(checkpoint.parameters, checkpoint.model_state, device)
        self._print("creating optimizer...")
        optimizer = self.create_optimizer(model, checkpoint.parameters, checkpoint.optimizer_state)
        self._print("creating dataloader...")
        batches = DataLoader(dataset = self.train_data, batch_size = self.batch_size, shuffle = False)
        num_batches = len(batches)
        # reset loss dict
        checkpoint.loss[self.LOSS_GROUP] = dict.fromkeys(self.loss_functions, 0.0)
        self._print("Training...")
        for batch_index, (x, y) in enumerate(batches):
            self._print(f"({batch_index + 1}/{num_batches})", end=" ")
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
        checkpoint.time[self.LOSS_GROUP] = float(time.time_ns() - start_train_time_ns) * float(10**(-9))