import os
import math
import operator
import random
import copy
import torch
import time
from hyperparameters import Hyperparameter
from torch.utils.data import DataLoader
from database import Checkpoint, SharedDatabase
from utils import get_datetime_string

mp = torch.multiprocessing.get_context('spawn')

class PBTController(mp.Process):
    def __init__(self, controller, evolve_queue, train_queue, database, frequency = 5, max_steps_criterion = None, max_epochs_criterion = None, score_criterion = None):
        assert not frequency or isinstance(frequency, int) and frequency > 0, f"Frequency must be of type {int} as 1 or higher."
        assert not max_steps_criterion or isinstance(max_steps_criterion, int) and max_steps_criterion > 0, f"The max steps criterion must be of type {int} as 1 or higher."
        assert not max_epochs_criterion or isinstance(max_epochs_criterion, int) and max_epochs_criterion > 0, f"The max epochs criterion must be of type {int} as 1 or higher."
        assert not score_criterion or isinstance(score_criterion, float), f"The score criterion must be of type {float}."
        self.controller = controller
        self.evolve_queue = evolve_queue
        self.train_queue = train_queue
        self.database = database
        self.frequency = frequency
        self.max_steps_criterion = max_steps_criterion
        self.max_epochs_criterion = max_epochs_criterion
        self.score_criterion = score_criterion

    def run(self):
        # TODO:m
        # create and queue checkpoints
        while True:
            if self.evolve_queue.empty():
                checkpoint = self.evolve_queue.get()
                if checkpoint.steps >= self.max_steps_criterion or checkpoint.epochs >= self.max_epochs_criterion:
                    continue
                if checkpoint.score >= self.score_criterion:
                    break
                if not self.is_ready(checkpoint):
                    self.train_queue.put(checkpoint)
                    continue
                self.controller.evolve(checkpoint)
                self.train_queue.put(checkpoint)

    def is_ready(self, checkpoint):
        """True every n-th epoch."""
        return checkpoint.steps % self.frequency == 0

    def is_finished(self, checkpoint):
        """ True if a given number of epochs have passed or if the score reaches above 99%. """
        if 'epochs' in self.end_criteria and checkpoint.epochs >= self.end_criteria['epochs']:
            # the number of epochs is equal or above the given treshold
            return True
        if 'steps' in self.end_criteria and checkpoint.steps >= self.end_criteria['steps']:
            # the number of steps is equal or above the given treshold
            return True
        if 'score' in self.end_criteria and checkpoint.score >= self.end_criteria['score']:
            # score is above the given treshold
            return True
        return False