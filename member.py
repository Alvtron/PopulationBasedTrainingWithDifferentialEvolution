import os
import copy
import torch
import math
from torch.utils.data import DataLoader
from database import Checkpoint, SharedDatabase
from utils import get_datetime_string

mp = torch.multiprocessing.get_context('spawn')

def create_model(model_class, model_state = None, device = 'cpu'):
        model = model_class().to(device)
        if model_state:
            model.load_state_dict(model_state)
        return model

def create_optimizer(model, optimizer_class, hyper_parameters, optimizer_state = None):
        optimizer = optimizer_class(model.parameters(), **hyper_parameters.get_optimizer_value_dict())
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        return optimizer

class Trainer(object):
    """ """
    def __init__(self, model_class, optimizer_class, loss_function, batch_size, train_data, device, verbose = False):
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.train_data = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = False)
        self.train_data_iterator = None
        self.device = device
        self.verbose = verbose
        self.steps = 0
        self.epochs = 0

    def train(self, hyper_parameters, model_state = None, optimizer_state = None, num_steps = 1):
        assert 0 < num_steps, "The number of steps must be at least one or higher."
        model = create_model(self.model_class, model_state, self.device)
        model.apply_hyper_parameters(hyper_parameters.model)
        optimizer = create_optimizer(model, self.optimizer_class, hyper_parameters, optimizer_state)
        if not self.train_data_iterator:
            self.train_data_iterator = iter(self.train_data)
        for _ in range(num_steps):
            try:
                x, y = next(self.train_data_iterator)
                self.step(model, optimizer, x, y)
                self.steps += 1
            except StopIteration:
                self.epochs += 1
                self.train_data_iterator = iter(self.train_data)
        return model.state_dict(), optimizer.state_dict()

    def step(self, model, optimizer, x, y):
        model.train()
        x, y = x.to(self.device), y.to(self.device)
        output = model(x)
        loss = self.loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class Evaluator(object):
    """ """
    def __init__(self, model_class, batch_size, test_data, device, verbose = False):
        self.model_class = model_class
        self.batch_size = batch_size
        self.test_data = DataLoader(test_data, self.batch_size, True)
        self.device = device
        self.verbose = verbose

    def eval(self, model_state):
        """Evaluate model on the provided validation or test set."""
        model = create_model(self.model_class, model_state, self.device)
        correct = 0
        for x, y in self.test_data:
            x, y = x.to(self.device), y.to(self.device)
            output = model(x)
            prediction = output.argmax(1)
            correct += prediction.eq(y).sum().item()
        accuracy = 100. * correct / (len(self.test_data) * self.batch_size)
        return accuracy

class Member(mp.Process):
    """A individual member in the population"""
    def __init__(self, id, model_class, optimizer_class, controller, batch_size, hyper_parameters, loss_function, train_data, test_data, database, device, verbose = 0, logging = False):
        super().__init__()
        self.id = id
        self.trainer = Trainer(model_class, optimizer_class, loss_function, batch_size, train_data, device, verbose)
        self.evaluator = Evaluator(model_class, batch_size, test_data, device, verbose)
        self.model_state = None
        self.optimizer_state = None
        self.hyper_parameters = hyper_parameters
        self.controller = controller
        self.database = database
        self.device = device
        self.verbose = verbose
        self.logging = logging
        self.score = None

    def __str__(self):
        num_training_batches = len(self.trainer.train_data)
        num_steps_this_epoch = self.trainer.steps % num_training_batches
        return f"{get_datetime_string()} - epoch {self.trainer.epochs} ({num_steps_this_epoch}/{num_training_batches}) - member {self.id}"

    def log(self, message):
        """Logs and prints the provided message in the appropriate syntax."""
        full_message = f"{self}: {message}"
        # TODO: Log message to file
        # if logging: self.database.log(id, message)
        if self.verbose > 0: print(full_message)

    def run(self):
        # prepare hyper-parameters with controller
        self.controller.prepare(self.hyper_parameters, self.log)
        # run population-based training loop
        while not self.controller.is_finished(self.create_checkpoint(), self.database): # not end of training
            # peform a training step
            #self.log("training...")
            self.model_state, self.optimizer_state = self.trainer.train(self.hyper_parameters, self.model_state, self.optimizer_state)
            if self.controller.is_ready(self.create_checkpoint(), self.database): # if ready-condition
                self.log("evaluating...")
                self.score = self.evaluator.eval(self.model_state)
                self.log(f"{self.score:.4f}%")
                # save to population
                self.save_checkpoint()
                # evolve member
                self.log("evolving...")
                self.model_state, self.optimizer_state, self.hyper_parameters = self.controller.evolve(
                    self.create_checkpoint(),
                    self.database,
                    self.trainer,
                    self.evaluator,
                    self.log)
                #self.score = self.evaluator.eval(self.model_state)
        self.log("finished.")
     
    def save_checkpoint(self):
        """Save checkpoint from member state."""
        checkpoint = self.create_checkpoint()
        self.database.save_entry(checkpoint)

    def create_checkpoint(self):
        """Create checkpoint from member state."""
        checkpoint = Checkpoint(
            self.id,
            self.trainer.epochs,
            self.trainer.steps,
            self.model_state,
            self.optimizer_state,
            self.hyper_parameters,
            self.score)
        return checkpoint

    def load_checkpoint(self, checkpoint):
        """Load member state from checkpoint object."""
        self.id = checkpoint.id
        self.epochs = checkpoint.epochs
        self.steps = checkpoint.stepsæå
        self.score = checkpoint.score
        self.hyper_parameters = checkpoint.hyper_parameters
        # Load model and optimizer state
        self.model_state = checkpoint.model_state
        self.optimizer_state = checkpoint.optimizer_state
