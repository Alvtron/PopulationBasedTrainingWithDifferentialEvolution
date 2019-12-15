import argparse
import os
import math
import torch
import torchvision
import torch.utils.data
import math
import operator
import random
import copy
from functools import partial 
from database import Checkpoint
from utils import get_datetime_string
from member import Member

mp = torch.multiprocessing.get_context('spawn')

class Controller(object):
    def __init__(self, population_size, hyper_parameters, trainer, evaluator, tester, evolver, database, step_size = 1, evolve_frequency = 5, end_criteria = {'score': 100.0}, device = 'cpu', verbose=True, logging=True):
        assert evolve_frequency and isinstance(evolve_frequency, int) and evolve_frequency > 0, f"Frequency must be of type {int} as 1 or higher."
        assert end_criteria and isinstance(end_criteria, dict), f"End criteria must be of type {dict}."
        self.population_size = population_size
        self.hyper_parameters = hyper_parameters
        self.trainer = trainer
        self.evaluator = evaluator
        self.tester = tester
        self.evolver = evolver
        self.database = database
        self.step_size = step_size
        self.evolve_frequency = evolve_frequency
        self.end_criteria = end_criteria
        self.device = device
        self.verbose = verbose
        self.logging = logging
        # create shared bool-value for when to end training
        self.end_event = torch.multiprocessing.Event()
        # create queues for training and evolving
        self.train_queue = torch.multiprocessing.Queue(population_size)
        self.evolve_queue = torch.multiprocessing.Queue(population_size)
        self.finish_queue = torch.multiprocessing.Queue(population_size)
        self.__workers = []

    def log(self, checkpoint, message):
        """Logs and prints the provided message in the appropriate syntax."""
        full_message = f"{checkpoint}: {message}"
        if self.logging:
            log_file_name = f"{checkpoint.id:03d}_log.txt"
            self.database.save_to_file(log_file_name, full_message)
        if self.verbose:
            print(full_message)

    def eval_function(self, checkpoint):
        current_model_state, _, _, _ , _ = self.trainer.train(
            hyper_parameters=checkpoint.hyper_parameters,
            model_state=checkpoint.model_state,
            optimizer_state=checkpoint.optimizer_state,
            epochs=checkpoint.epochs,
            steps=checkpoint.steps)
        score = self.evaluator.eval(current_model_state)
        return score

    def is_member_ready(self, checkpoint):
        """True every n-th epoch."""
        return checkpoint.steps % self.evolve_frequency == 0

    def is_member_finished(self, checkpoint):
        if 'epochs' in self.end_criteria and checkpoint.epochs >= self.end_criteria['epochs']:
            # the number of epochs is equal or above the given treshold
            return True
        if 'steps' in self.end_criteria and checkpoint.steps >= self.end_criteria['steps']:
            # the number of steps is equal or above the given treshold
            return True
        return False
    
    def is_population_finished(self, checkpoint):
        if 'score' in self.end_criteria and checkpoint.eval_score >= self.end_criteria['score']:
            # score is above the given treshold
            return True
        return False

    def start_training_procedure(self):
        # start workers
        self.__workers = [
            Member(
                end_event = self.end_event,
                trainer = self.trainer,
                evaluator = self.evaluator,
                evolve_queue = self.evolve_queue,
                train_queue = self.train_queue,
                step_size = self.step_size,
                device = self.device,
                verbose = self.verbose)
            for _ in range(self.population_size)]
        [w.start() for w in self.__workers]
        # queue checkpoints
        for id in range(self.population_size):
            hyper_parameters = copy.deepcopy(self.hyper_parameters)
            # prepare hyper_parameters
            for _, hyper_parameter in hyper_parameters:
                hyper_parameter.sample_uniform()
            checkpoint = Checkpoint(id, hyper_parameters)
            self.train_queue.put(checkpoint)

    def start(self):
        try:
            # start training
            print("Starting worker processess...")
            self.start_training_procedure()
            # controller loop
            while not self.end_event.is_set():
                if self.evolve_queue.empty():
                    continue
                # get next checkpoint
                checkpoint = self.evolve_queue.get()
                # save checkpoint to database
                self.database.save_entry(checkpoint)
                if self.is_population_finished(checkpoint):
                    self.log(checkpoint, "End criterium reached.")
                    self.end_event.set()
                    break
                if self.is_member_finished(checkpoint):
                    self.log(checkpoint, "finished.")
                    self.finish_queue.put(checkpoint)
                    if self.finish_queue.full():
                        print("All workers are finished.")
                        self.end_event.set()
                        break
                    else: continue
                # evolve member if ready
                if self.is_member_ready(checkpoint):
                    self.log(checkpoint, "evolving...")
                    self.evolver.evolve(
                        member=checkpoint,
                        generation=self.database.get_latest,
                        population=self.database.get_all,
                        function=self.eval_function,
                        logger=partial(self.log, checkpoint))
                self.log(checkpoint, "training...")
                self.train_queue.put(checkpoint)
            # terminate worker processes
            print("Controller is finished.")
        except KeyboardInterrupt:
            print("Controller was interupted.")
        finally:
            print("Terminating all left-over worker-processes...")
            [worker.terminate() for worker in self.__workers]
            print("Termination was successfull!")