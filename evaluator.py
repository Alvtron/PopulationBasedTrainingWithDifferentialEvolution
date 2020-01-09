import torch
from torch.utils.data import DataLoader
        
class Evaluator(object):
    """ Class for evaluating the performance of the provided model on the set evaluation dataset. """
    def __init__(self, model_class, eval_metrics, batch_size, test_data, device, load_in_memory=False, verbose = False):
        self.model_class = model_class
        self.eval_metrics = eval_metrics
        self.batch_size = batch_size
        self.test_data = DataLoader(
            dataset=test_data,
            batch_size = batch_size,
            shuffle = False)
        if load_in_memory: self.test_data = list(self.test_data)
        self.device = device
        self.verbose = verbose

    def create_model(self, model_state = None):
        model = self.model_class().to(self.device)
        if model_state:
            model.load_state_dict(model_state)
        return model

    def eval(self, model_state):
        """Evaluate model on the provided validation or test set."""
        model = self.create_model(model_state)
        dataset_length = len(self.test_data)
        metric_values = dict.fromkeys(self.eval_metrics, 0.0)
        for x, y in self.test_data:
            x, y = x.to(self.device), y.to(self.device)
            output = model(x)
            # compute losses
            for metric_type, metric_function in self.eval_metrics.items():
                metric_values[metric_type] += metric_function(output, y).item() / dataset_length
        return metric_values