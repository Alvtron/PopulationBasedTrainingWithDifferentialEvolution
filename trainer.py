import torch
import itertools
from torch.utils.data import DataLoader

class Trainer(object):
    """ """
    def __init__(self, model_class, optimizer_class, loss_function, batch_size, train_data, device, verbose = False):
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.train_data = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = False)
        self.device = device
        self.verbose = verbose
        self.steps = 0
        self.epochs = 0

    def create_model(self, model_state = None):
        model = self.model_class().to(self.device)
        if model_state:
            model.load_state_dict(model_state)
        return model

    def create_optimizer(self, model, hyper_parameters, optimizer_state = None):
        optimizer = self.optimizer_class(model.parameters(), **hyper_parameters.get_optimizer_value_dict())
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        return optimizer

    def step(self, model, optimizer, x, y):
        model.train()
        x, y = x.to(self.device), y.to(self.device)
        output = model(x)
        loss = self.loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, hyper_parameters, model_state, optimizer_state, epochs, steps, step_size = 1):
        if step_size < 1:
            raise Exception("The number of steps must be at least one or higher.")
        # creating iterator
        num_steps_this_epoch = steps % len(self.train_data)
        iterator = itertools.islice(self.train_data, num_steps_this_epoch, None)
        # preparing model and optimizer
        model = self.create_model(model_state)
        model.apply_hyper_parameters(hyper_parameters.model)
        optimizer = self.create_optimizer(model, hyper_parameters, optimizer_state)
        steps_left = step_size
        total_loss = 0
        while steps_left > 0:
            try:
                x, y = next(iterator)
                total_loss += self.step(model, optimizer, x, y)
                steps_left -= 1
                steps += 1
                if self.verbose: print(f"Epochs: {self.epochs}, steps: {self.steps}, loss: {loss}")
            except StopIteration:
                iterator = iter(self.train_data)
                epochs += 1
        avg_loss = total_loss / step_size
        return model.state_dict(), optimizer.state_dict(), epochs, steps, avg_loss