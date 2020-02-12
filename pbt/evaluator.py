from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module

from .hyperparameters import Hyperparameters

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.multiprocessing.set_sharing_strategy('file_descriptor')

class Evaluator(object):
    """ Class for evaluating the performance of the provided model on the set evaluation dataset. """
    def __init__(
            self, model_class : Module, test_data : Dataset, batch_size : int, loss_functions : dict,
            device : str, verbose : bool = False):
        self.model_class = model_class
        self.test_data = DataLoader(
            dataset=test_data,
            batch_size = batch_size,
            shuffle = False,
            num_workers=0)
        self.batch_size = batch_size
        self.loss_functions = loss_functions
        self.device = device
        self.verbose = verbose

    def create_model(self, model_state = None):
        model = self.model_class().to(self.device)
        if model_state:
            model.load_state_dict(model_state)
        model.eval()
        return model

    def eval(self, model_state : dict):
        """Evaluate model on the provided validation or test set."""
        dataset_length = len(self.test_data)
        metric_values = dict.fromkeys(self.loss_functions, 0.0)
        with torch.cuda.device(0):
            model = self.create_model(model_state)
            for batch_index, (x, y) in enumerate(self.test_data, start=1):
                if self.verbose: print(f"({batch_index}/{dataset_length})", end=" ")
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.no_grad():
                    output = model(x)
                for metric_type, metric_function in self.loss_functions.items():
                    with torch.no_grad():
                        loss = metric_function(output, y)
                    metric_values[metric_type] += loss.item() / float(dataset_length)
                    if self.verbose: print(f"{metric_type}: {loss.item():4f}", end=" ")
                    del loss
                if self.verbose: print(end="\n")
                del output
            del model
        torch.cuda.empty_cache()
        return metric_values