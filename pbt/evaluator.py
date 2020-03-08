import time
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module

from .hyperparameters import Hyperparameters

class Evaluator(object):
    """ Class for evaluating the performance of the provided model on the set evaluation dataset. """
    def __init__(
            self, model_class : Module, test_data : Dataset, batch_size : int, loss_functions : dict, verbose : bool = False):
        self.model_class = model_class
        self.test_data = DataLoader(
            dataset=test_data,
            batch_size = batch_size,
            shuffle = False,
            num_workers=0)
        self.batch_size = batch_size
        self.loss_functions = loss_functions
        self.verbose = verbose

    def _print(self, message : str = None, end : str = '\n'):
        if not self.verbose:
            return
        print(message, end=end)

    def create_model(self, model_state = None, device : str = 'cpu'):
        self._print("creating model...")
        model = self.model_class().to(device)
        if model_state:
            model.load_state_dict(model_state)
        model.eval()
        return model

    def _evaluate(self, checkpoint : dict, device : str):
        """Evaluate model on the provided validation or test set."""
        start_eval_time_ns = time.time_ns()
        dataset_length = len(self.test_data)
        checkpoint.loss['eval'] = dict.fromkeys(self.loss_functions, 0.0)
        model = self.create_model(checkpoint.model_state, device)
        for batch_index, (x, y) in enumerate(self.test_data, start=1):
            if self.verbose: print(f"({batch_index}/{dataset_length})", end=" ")
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                output = model(x)
            for metric_type, metric_function in self.loss_functions.items():
                with torch.no_grad():
                    loss = metric_function(output, y)
                checkpoint.loss['eval'][metric_type] += loss.item() / float(dataset_length)
                if self.verbose: print(f"{metric_type}: {loss.item():4f}", end=" ")
                del loss
            if self.verbose: print(end="\n")
            del output
        # clean GPU memory
        del model
        torch.cuda.empty_cache()
        # update checkpoint
        checkpoint.time['eval'] = float(time.time_ns() - start_eval_time_ns) * float(10**(-9))

    def _evaluate_on_gpu(self, checkpoint : dict, device : str):
        with torch.cuda.device(device):
            self._evaluate(checkpoint, device)

    def __call__(self, checkpoint : dict, device : str = 'cpu'):
        if device == 'cpu':
            self._evaluate(checkpoint, device)
        elif device.startswith('cuda'):
            self._evaluate_on_gpu(checkpoint, device)
        else:
            raise NotImplementedError(f"Device '{device}' is not supported.")