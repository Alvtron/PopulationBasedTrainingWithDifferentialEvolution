import torch
import itertools
from hyperparameters import Hyperparameters
from torch.utils.data import DataLoader

class Trainer(object):
    """ A class for training the provided model with the provided hyper-parameters on the set training dataset. """
    def __init__(self, model_class, optimizer_class, loss_metric, eval_metrics, batch_size, train_data, device, load_in_memory=True, verbose = False):
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.loss_metric = loss_metric
        self.eval_metrics = eval_metrics
        self.batch_size = batch_size
        self.train_data = DataLoader(
            dataset = train_data,
            batch_size = batch_size,
            shuffle = False)
        if load_in_memory: self.train_data = list(self.train_data)
        self.load_in_memory = load_in_memory
        self.device = device
        self.verbose = verbose

    def create_model(self, model_state = None):
        model = self.model_class().to(self.device)
        if model_state:
            model.load_state_dict(model_state)
        return model

    def create_optimizer(self, model, hyper_parameters : Hyperparameters, optimizer_state = None):
        optimizer = self.optimizer_class(model.parameters(), **hyper_parameters.get_optimizer_value_dict())
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
            for param_name, param_value in hyper_parameters.optimizer.items():
                for param_group in optimizer.param_groups:
                    param_group[param_name] = param_value.value
        return optimizer

    def train(self, hyper_parameters, model_state, optimizer_state, epochs, steps, step_size = 1):
        if step_size < 1:
            raise Exception("The number of steps must be at least one or higher.")
        # preparing model and optimizer
        model = self.create_model(model_state)
        model.apply_hyper_parameters(hyper_parameters.model, self.device)
        model.train()
        optimizer = self.create_optimizer(model, hyper_parameters, optimizer_state)
        # creating iterator
        batch_index = steps % len(self.train_data)
        dataset_iterator = itertools.islice(self.train_data, batch_index, None)
        # initialize eval metrics dict
        metric_values = dict.fromkeys(self.eval_metrics, 0.0)
        # loop until step_size is exhausted
        END_STEPS = steps + step_size
        while steps != END_STEPS:
            try:
                x, y = next(dataset_iterator)
                x, y = x.to(self.device), y.to(self.device)
                for metric_type, metric_function in self.eval_metrics.items():
                    if metric_type == self.loss_metric:
                        output = model(x)
                        loss = metric_function(output, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        metric_values[metric_type] += loss.item() / step_size
                    else:
                        metric_values[metric_type] += metric_function(output, y).item() / step_size
                steps += 1
            except StopIteration:
                dataset_iterator = iter(self.train_data)
                epochs += 1
        return model.state_dict(), optimizer.state_dict(), epochs, steps, metric_values