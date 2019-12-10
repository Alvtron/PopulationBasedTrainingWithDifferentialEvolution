import torch
from torch.utils.data import DataLoader
        
class Evaluator(object):
    """ """
    def __init__(self, model_class, batch_size, test_data, device, verbose = False):
        self.model_class = model_class
        self.batch_size = batch_size
        self.test_data = DataLoader(test_data, self.batch_size, True)
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
        correct = 0
        for x, y in self.test_data:
            x, y = x.to(self.device), y.to(self.device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1) # argmax, ex. [0.1, 0.3, 0.5, 0.1] --> 2th index
            correct += predicted.eq(y).sum().item() # add 1 for each correct predict of the batch
        dataset_length = len(self.test_data) * self.batch_size
        accuracy = 100. * (correct / dataset_length)
        if self.verbose: print(f"Accuracy: {accuracy:.4f}%")
        return accuracy