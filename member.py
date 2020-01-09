import os
import copy
import torch
import math
import time
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader
from database import Checkpoint, SharedDatabase
from utils import get_datetime_string

mp = torch.multiprocessing.get_context('spawn')

class Member(mp.Process):
    """A individual member in the population"""
    def __init__(self, end_event, evolve_queue, train_queue, trainer, evaluator, step_size = 1, device = "cpu", verbose = False):
        super().__init__()
        self.end_event = end_event
        self.evolve_queue = evolve_queue
        self.train_queue = train_queue
        self.trainer = trainer
        self.evaluator = evaluator
        self.step_size = step_size
        self.device = device
        self.verbose = verbose

    def run(self):
        while not self.end_event.is_set():
            if self.train_queue.empty():
                continue
            # get next checkpoint from train queue
            checkpoint = self.train_queue.get()
            # train checkpoint model
            start_train_time_ns = time.time_ns()
            checkpoint.model_state, checkpoint.optimizer_state, checkpoint.epochs, checkpoint.steps, checkpoint.loss['train'] = self.trainer.train(
                hyper_parameters=checkpoint.hyper_parameters,
                model_state=checkpoint.model_state,
                optimizer_state=checkpoint.optimizer_state,
                epochs=checkpoint.epochs,
                steps=checkpoint.steps,
                step_size=self.step_size)
            checkpoint.time['train'] = float(time.time_ns() - start_train_time_ns) * float(10**(-9))
            # evaluate checkpoint model
            start_eval_time_ns = time.time_ns()
            checkpoint.loss['eval'] = self.evaluator.eval(checkpoint.model_state)
            checkpoint.time['eval'] = float(time.time_ns() - start_eval_time_ns) * float(10**(-9))
            self.evolve_queue.put(checkpoint)