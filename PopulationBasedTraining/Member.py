import os
import torch
from torch.utils.data import DataLoader
from database import Checkpoint, Database
from utils import get_datetime_string
from controller import Controller
from hyperparameters import hyperparameters_to_value_dict

mp = torch.multiprocessing.get_context('spawn')

class Member(mp.Process):
    '''A individual member in the population'''
    def __init__(self, id, model, optimizer_class, controller, hyperparameters, loss_function, train_data, test_data, database, device, verbose):
        super().__init__()
        self.id = id
        self.model = model
        self.optimizer_class = optimizer_class
        self.controller = controller
        self.hyperparameters = hyperparameters
        self.loss_function = loss_function
        self.train_data = train_data
        self.test_data = test_data
        self.database = database
        self.device = device
        self.verbose = verbose
        self._epoch = 1
        self._score = None
        self._is_mutated = False

    def __str__(self):
        return f"{get_datetime_string()} - epoch {self._epoch} - member {self.id}"

    def _log(self, message):
        if self.verbose > 0: print (f"{self}: {message}")

    def run(self):
        # prepare hyper-parameters with controller
        self.controller.prepare(self.hyperparameters, self._log)
        # create optimizer
        optimizer_parameters = hyperparameters_to_value_dict(self.hyperparameters['optimizer'])
        self.optimizer = self.optimizer_class(self.model.parameters(), **optimizer_parameters)
        # run population-based training loop
        while not self.controller.is_finished(self.create_checkpoint(), self.database): # not end of training
            self.train() # step
            self._score = self.eval() # eval
            if self.controller.is_ready(self.create_checkpoint(), self.database): # if ready-condition
                self.mutate() # mutation
                self._score = self.eval()
                self._is_mutated = True
            else:
                self._is_mutated = False
            self._log(f"{self._score:.2f}%")
            # save to population
            self.save_checkpoint()
            self._epoch += 1
        self._log("finished.")

    def mutate(self):
        checkpoint = self.create_checkpoint()
        self.controller.evolve(checkpoint, self.database, self._log)
        self.load_checkpoint(checkpoint)

    def train(self):
        """Train the model on the provided training set."""
        self._log("training...")
        self.model.train()
        dataloader = DataLoader(self.train_data, self.hyperparameters['batch_size'].value(), True)
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.loss_function(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self):
        """Evaluate model on the provided validation or test set."""
        self._log(f"evaluating...")
        self.model.eval()
        dataloader = DataLoader(self.test_data, self.hyperparameters['batch_size'].value(), True)
        correct = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            pred = output.argmax(1)
            correct += pred.eq(y).sum().item()
        accuracy = 100. * correct / (len(dataloader) * self.hyperparameters['batch_size'].value())
        return accuracy
     
    def save_checkpoint(self):
        """Save checkpoint from member state."""
        checkpoint = self.create_checkpoint()
        self.database.save_entry(checkpoint)

    def create_checkpoint(self):
        """Create checkpoint from member state."""
        checkpoint = Checkpoint(
            self.id,
            self._epoch,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.hyperparameters,
            self._score,
            self._is_mutated)
        return checkpoint

    def load_checkpoint(self, checkpoint):
        """Load member state from checkpoint object."""
        self.id = checkpoint.id
        self._epoch = checkpoint.epoch
        self._score = checkpoint.score
        self._is_mutated = checkpoint.is_mutated
        self.hyperparameters = checkpoint.hyperparameters
        # Load model and optimizer state
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        # apply hyperparameters to model and optimizer
        self.apply_hyperparameters()
        # set dropout and batch normalization layers to evaluation mode before running inference
        self.model.eval()

    def apply_hyperparameters(self):
        for hyperparameter_name, hyperparameter in self.hyperparameters['optimizer'].items():
            for param_group in self.optimizer.param_groups:
                # create a random perturbation factor with the given perturb factors
                param_group[hyperparameter_name] = hyperparameter.value()