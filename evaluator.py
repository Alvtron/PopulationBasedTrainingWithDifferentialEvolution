import torch
from torch.utils.data import DataLoader

from hyperparameters import Hyperparameters
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module

class Evaluator(object):
    """ Class for evaluating the performance of the provided model on the set evaluation dataset. """
    def __init__(
            self, model_class : Module, test_data : Dataset, batch_size : int, loss_functions : dict,
            device : str, load_in_memory : bool = False, verbose : bool = False):
        self.model_class = model_class
        self.test_data = DataLoader(
            dataset=test_data,
            batch_size = batch_size,
            shuffle = False)
        if load_in_memory: self.test_data = list(self.test_data)
        self.batch_size = batch_size
        self.loss_functions = loss_functions
        self.device = device
        self.verbose = verbose

    def create_model(self, model_state = None):
        model = self.model_class().to(self.device)
        if model_state:
            model.load_state_dict(model_state)
        return model

    def eval(self, model_state : dict):
        """Evaluate model on the provided validation or test set."""
        model = self.create_model(model_state)
        dataset_length = len(self.test_data)
        metric_values = dict.fromkeys(self.loss_functions, 0.0)
        for batch_index, (x, y) in enumerate(self.test_data, start=1):
            if self.verbose: print(f"({batch_index}/{dataset_length})", end=" ")
            x, y = x.to(self.device), y.to(self.device)
            output = model(x)
            for metric_type, metric_function in self.loss_functions.items():
                loss = metric_function(output, y)
                metric_values[metric_type] += loss.item() / float(dataset_length)
                if self.verbose: print(f"{metric_type}: {loss.item():4f}", end=" ")
            if self.verbose: print(end="\n")
        return metric_values