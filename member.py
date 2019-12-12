import os
import copy
import torch
import math
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader
from database import Checkpoint, SharedDatabase
from utils import get_datetime_string

mp = torch.multiprocessing.get_context('spawn')

class Member(mp.Process):
    """A individual member in the population"""
    def __init__(self, id, controller, hyper_parameters, trainer, evaluator, database, step_size = 1, device = "cpu", verbose = 0, logging = False):
        super().__init__()
        self.id = id
        self.trainer = trainer
        self.evaluator = evaluator
        self.model_state = None
        self.optimizer_state = None
        self.hyper_parameters = hyper_parameters
        self.controller = controller
        self.database = database
        self.step_size = step_size
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
        if self.logging:
            log_file_name = f"{self.id:03d}_log.txt"
            self.database.save_to_file(log_file_name, full_message)
        if self.verbose:
            print(full_message)

    def run(self):
        # prepare hyper-parameters with controller
        self.controller.prepare(self.hyper_parameters, self.log)
        # run population-based training loop
        while not self.controller.is_finished(self.create_checkpoint(), self.database): # not end of training
            # peform a training step
            #self.log("training...")
            self.model_state, self.optimizer_state = self.trainer.train(self.hyper_parameters, self.model_state, self.optimizer_state, self.step_size)
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