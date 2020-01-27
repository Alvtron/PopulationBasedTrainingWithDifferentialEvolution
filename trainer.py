import torch
import torchvision
import itertools
import matplotlib.pyplot as plt
from hyperparameters import Hyperparameters
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.vision import StandardTransform
from torch.optim import Optimizer
from models import HyperNet
from copy import deepcopy
from utils.data import create_subset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.multiprocessing.set_sharing_strategy('file_descriptor')

class Trainer(object):
    """ A class for training the provided model with the provided hyper-parameters on the set training dataset. """
    def __init__(
            self, model_class : HyperNet, optimizer_class : Optimizer, train_data : Dataset, batch_size : int,
            loss_functions : dict, loss_metric : str, device : str, verbose : bool = False):
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.train_data = train_data
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

    def create_subset(self, steps, hyper_parameters):
        dataset_size = len(self.train_data)
        start_index = (steps * self.batch_size) % dataset_size
        subset = create_subset(self.train_data, start_index, dataset_size)
        subset.update(hyper_parameters.augment)
        return iter(DataLoader(
            dataset = subset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers=0))

    def train(
            self, hyper_parameters : Hyperparameters, model_state : dict, optimizer_state : dict,
            epochs : int, steps : int, step_size : int = 1):
        if step_size < 1:
            raise ValueError("The number of steps must be at least one or higher.")
        with torch.cuda.device(0):
            # preparing model and optimizer
            model = self.create_model(model_state)
            model.apply_hyper_parameters(hyper_parameters.model, self.device)
            model.train()
            optimizer = self.create_optimizer(model, hyper_parameters, optimizer_state)
            # create subset
            batches = self.create_subset(steps, hyper_parameters)
            # initialize eval metrics dict
            metric_values = dict.fromkeys(self.loss_functions, 0.0)
            # loop until step_size is exhausted
            END_STEPS = steps + step_size
            while steps != END_STEPS:
                # creating iterator
                try:
                    if self.verbose: print(f"({1 + step_size - (END_STEPS - steps)}/{step_size})", end=" ")
                    x, y = next(batches)
                    #grid_img = torchvision.utils.make_grid(x, nrow=8)
                    #plt.imshow(grid_img.permute(1, 2, 0))
                    #plt.show()
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
                    batches = self.create_subset(0, hyper_parameters)
                    epochs += 1
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
        torch.cuda.empty_cache()
        return model_state, optimizer_state, epochs, steps, metric_values