import torch
from torch.utils.data import DataLoader

class Trainer(object):
    """ """
    def __init__(self, model_class, optimizer_class, loss_function, batch_size, train_data, device, verbose = False):
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.train_data = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = False)
        self.iterator = None
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

    def train(self, hyper_parameters, model_state = None, optimizer_state = None, num_steps = 1, register = True):
        if num_steps < 1:
            raise Exception("The number of steps must be at least one or higher.")
        # creating iterator if not created
        if not self.iterator:
            self.iterator = iter(self.train_data)
        # preparing model and optimizer
        model = self.create_model(model_state)
        model.apply_hyper_parameters(hyper_parameters.model)
        optimizer = self.create_optimizer(model, hyper_parameters, optimizer_state)
        steps_left = num_steps
        while steps_left > 0:
            try:
                x, y = next(self.iterator)
                self.step(model, optimizer, x, y)
                steps_left -= 1
                if register: self.steps += 1
            except StopIteration:
                self.iterator = iter(self.train_data)
                if register: self.epochs += 1
        return model.state_dict(), optimizer.state_dict()