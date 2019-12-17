import os
import math
import operator
import random
import copy
import argparse
import time
import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
from functools import partial 
from database import Checkpoint
from member import Member
from utils import get_datetime_string

mp = torch.multiprocessing.get_context('spawn')

class Controller(object):
    def __init__(self, population_size, hyper_parameters, trainer, evaluator, tester, evolver, database, tensorboard_writer = None, step_size = 1, evolve_frequency = 5, end_criteria = {'score': 100.0}, device = 'cpu', verbose=True, logging=True):
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
        self.end_event = mp.Event()
        # create queues for training and evolving
        self.train_queue = mp.Queue(population_size)
        self.evolve_queue = mp.Queue(population_size)
        self.finish_queue = mp.Queue(population_size)
        self.__workers = []
        self.__tensorboard_writer = tensorboard_writer
        self.__resource_distribution = {"init": 0, "tensorboard": 0, "evolve": 0}

    def log(self, checkpoint, message):
        """Logs and prints the provided message in the appropriate syntax."""
        time = get_datetime_string()
        full_message = f"{time} {checkpoint}: {message}"
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
        start_time = time.time()
        print("Creating worker processes...")
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
        # Starting workers
        for index, worker in enumerate(self.__workers, start=1):
            print(f"Starting worker {index} of {self.population_size}", end="\r", flush=True)
            worker.start()
        # queue checkpoints
        for id in range(self.population_size):
            print(f"Queuing checkpoint {id} of {self.population_size - 1}...")
            hyper_parameters = copy.deepcopy(self.hyper_parameters)
            # prepare hyper_parameters
            for _, hyper_parameter in hyper_parameters:
                hyper_parameter.sample_uniform()
            checkpoint = Checkpoint(id, hyper_parameters)
            self.train_queue.put(checkpoint)
        self.__resource_distribution["init"] += time.time() - start_time

    def write_to_tensorflow(self, checkpoint):
        """Plots loss and eval metric to tensorboard"""
        start_time = time.time()
        # train loss
        self.__tensorboard_writer.add_scalar(
            tag=f"Loss/train/{checkpoint.id:03d}",
            scalar_value=checkpoint.train_loss,
            global_step=checkpoint.steps)
        # eval score
        self.__tensorboard_writer.add_scalar(
            tag=f"Score/eval/{checkpoint.id:03d}",
            scalar_value=checkpoint.eval_score,
            global_step=checkpoint.steps)
        # hyper-parameters
        for hparam_name, hparam in checkpoint.hyper_parameters:
            self.__tensorboard_writer.add_scalar(
                tag=f"Hyperparameters/{hparam_name}/{checkpoint.id:03d}",
                scalar_value=hparam.value(),
                global_step=checkpoint.steps)
        # resource distribution
        self.__tensorboard_writer.add_scalars(
            main_tag=f"Resource_usage",
            tag_scalar_dict=self.__resource_distribution,
            global_step=checkpoint.steps)
        self.__resource_distribution["tensorboard"] += time.time() - start_time

    def start(self):
        try:
            # start training
            self.start_training_procedure()
            # controller loop
            while not self.end_event.is_set():
                if self.evolve_queue.empty():
                    continue
                # get next checkpoint
                checkpoint = self.evolve_queue.get()
                # save checkpoint to database
                self.database.save_entry(checkpoint)
                if self.__tensorboard_writer:
                    self.write_to_tensorflow(checkpoint)
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
                    start_time = time.time()
                    self.evolver.evolve(
                        member=checkpoint,
                        generation=self.database.get_latest,
                        population=self.database.get_all,
                        function=self.eval_function,
                        logger=partial(self.log, checkpoint))
                    self.__resource_distribution["evolve"] += time.time() - start_time
                self.log(checkpoint, "training...")
                self.train_queue.put(checkpoint)
            # terminate worker processes
            print("Controller is finished.")
        except KeyboardInterrupt:
            print("Controller was interupted.")
        finally:
            print("Terminating all left-over worker-processes...")
            for worker in self.__workers:
                if isinstance(worker, mp.Process):
                    worker.terminate()
                else: continue
            print("Termination was successfull!")