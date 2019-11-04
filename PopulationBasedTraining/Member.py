import os
import torch
from torch.utils.data import DataLoader
from database import Checkpoint, Database
from datetime import datetime

mp = torch.multiprocessing.get_context('spawn')

def get_datetime_string():
    date_and_time = datetime.now()
    return date_and_time.strftime('%Y-%m-%d %H:%M:%S')

class Member(mp.Process):
    '''A individual member in the population'''
    def __init__(self, id, model, optimizer, hyperparameters, mutation_function, ready_condition, end_training_condition, loss_function, train_data, test_data, database, device, verbose):
        super().__init__()
        self.id = id
        self.hyperparameters = hyperparameters
        self.mutation_function = mutation_function
        self.ready_condition = ready_condition
        self.end_training_condition = end_training_condition
        self.batch_size = hyperparameters['batch_size'].value
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr = 0.1, momentum = 0.9)
        self.loss_function = loss_function
        self.train_data = train_data
        self.test_data = test_data
        self.database = database
        self.device = device
        self.verbose = verbose
        self.epoch = 1
        self.score = None
        self.is_mutated = False

    def run(self):
        while not self.end_training_condition(self.create_checkpoint(), self.database): # not end of training
            self.train() # step
            self.score = self.eval() # eval
            if self.ready_condition(self.create_checkpoint(), self.database): # if ready-condition
                self.mutate() # mutation
                self.score = self.eval()
            else:
                self.is_mutated = False
            if self.verbose > 0:
                print(f"{get_datetime_string()} - epoch {self.epoch} - m{self.id}: scored {self.score:.2f}%")
            # save to population
            self.save_checkpoint()
            self.epoch += 1
        print(f"{get_datetime_string()} - epoch {self.epoch} - m{self.id}: finished.")

    def mutate(self):
        checkpoint = self.create_checkpoint()
        self.mutation_function(checkpoint, self.database)
        checkpoint.is_mutated = True
        self.apply_checkpoint(checkpoint)

    def train(self):
        """Train the model on the provided training set."""
        if self.verbose > 0:
            print(f"{get_datetime_string()} - epoch {self.epoch} - m{self.id}: training...")
        self.model.train()
        dataloader = DataLoader(self.train_data, self.batch_size, True)
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.loss_function(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self):
        """Evaluate model on the provided validation or test set."""
        if self.verbose > 0:
            print(f"{get_datetime_string()} - epoch {self.epoch} - m{self.id}: evaluating...")
        self.model.eval()
        dataloader = DataLoader(self.test_data, self.batch_size, True)
        correct = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            pred = output.argmax(1)
            correct += pred.eq(y).sum().item()
        accuracy = 100. * correct / (len(dataloader) * self.batch_size)
        return accuracy
     
    def save_checkpoint(self):
        """Save checkpoint from member state."""
        checkpoint = self.create_checkpoint()
        self.database.save_entry(checkpoint)

    def create_checkpoint(self):
        """Create checkpoint from member state."""
        checkpoint = Checkpoint(
            self.id,
            self.epoch,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.hyperparameters,
            self.batch_size,
            self.score,
            self.is_mutated)
        return checkpoint

    def apply_checkpoint(self, checkpoint):
        """Load member state from checkpoint object."""
        self.id = checkpoint.id
        self.epoch = checkpoint.epoch
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.batch_size = checkpoint.batch_size
        self.score = checkpoint.score
        self.is_mutated = checkpoint.is_mutated
        # set dropout and batch normalization layers to evaluation mode before running inference
        self.model.eval()