import torch
import itertools
from hyperparameters import Hyperparameters
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from models import HyperNet
from copy import deepcopy

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.multiprocessing.set_sharing_strategy('file_descriptor')

class Trainer(object):
    """ A class for training the provided model with the provided hyper-parameters on the set training dataset. """
    def __init__(
            self, model_class : HyperNet, optimizer_class : Optimizer, train_data : Dataset, batch_size : int,
            loss_functions : dict, loss_metric : str, device : str, load_in_memory : bool = True, verbose : bool = False):
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.train_data = DataLoader(
            dataset = train_data,
            batch_size = batch_size,
            shuffle = False,
            pin_memory=device.startswith('cuda'))
        if load_in_memory: self.train_data = list(self.train_data)
        self.batch_size = batch_size
        self.loss_functions = loss_functions
        self.loss_metric = loss_metric
        self.device = device
        self.verbose = verbose

    def create_model(self, model_state = None):
        model = self.model_class().to(self.device)
        if model_state:
            model.load_state_dict(model_state)
        return model

    def create_optimizer(self, model : HyperNet, hyper_parameters : Hyperparameters, optimizer_state : dict  = None):
        optimizer = self.optimizer_class(model.parameters(), **hyper_parameters.get_optimizer_value_dict())
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
            for param_name, param_value in hyper_parameters.optimizer.items():
                for param_group in optimizer.param_groups:
                    param_group[param_name] = param_value.value
        return optimizer

    def train(
            self, hyper_parameters : Hyperparameters, model_state : dict, optimizer_state : dict,
            epochs : int, steps : int, step_size : int = 1):
        if step_size < 1:
            raise Exception("The number of steps must be at least one or higher.")
        with torch.cuda.device(0):
            # preparing model and optimizer
            model = self.create_model(model_state)
            model.apply_hyper_parameters(hyper_parameters.model, self.device)
            model.train()
            optimizer = self.create_optimizer(model, hyper_parameters, optimizer_state)
            # creating iterator
            batch_index = steps % len(self.train_data)
            dataset_view = itertools.islice(self.train_data, batch_index, None)
            # initialize eval metrics dict
            metric_values = dict.fromkeys(self.loss_functions, 0.0)
            # loop until step_size is exhausted
            END_STEPS = steps + step_size
            while steps != END_STEPS:
                try:
                    if self.verbose: print(f"({1 + step_size - (END_STEPS - steps)}/{step_size})", end=" ")
                    x, y = next(dataset_view)
                    x, y = x.to(self.device), y.to(self.device)
                    for metric_type, metric_function in self.loss_functions.items():
                        if metric_type == self.loss_metric:
                            output = model(x)
                            model.zero_grad()
                            loss = metric_function(output, y)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            metric_values[metric_type] += loss.item() / float(step_size)
                        else:
                            loss = metric_function(output, y)
                            metric_values[metric_type] += loss.item() / float(step_size)
                        if self.verbose: print(f"{metric_type}: {loss.item():4f}", end=" ")
                    if self.verbose: print(end="\n")
                    steps += 1
                except StopIteration:
                    dataset_view = iter(self.train_data)
                    epochs += 1
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
        torch.cuda.empty_cache()
        return model_state, optimizer_state, epochs, steps, metric_values