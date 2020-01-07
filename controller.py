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

    def log(self, checkpoint, message):
        """Logs and prints the provided message in the appropriate syntax."""
        time = get_datetime_string()
        full_message = f"{time} {checkpoint}: {message}"
        if self.logging:
            log_file_name = f"{checkpoint.id:03d}_log.txt"
            self.database.append_to_file(tag='logs', file_name=log_file_name, text=full_message)
        if self.verbose:
            print(full_message)
        if self.__tensorboard_writer:
            self.__tensorboard_writer.add_text(tag="controller", text_string=full_message, global_step=checkpoint.steps)

    def eval_function(self, checkpoint):
        current_model_state, _, _, _ , _ = self.trainer.train(
            hyper_parameters=checkpoint.hyper_parameters,
            model_state=checkpoint.model_state,
            optimizer_state=checkpoint.optimizer_state,
            epochs=checkpoint.epochs,
            steps=checkpoint.steps,
            step_size=1
            )
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
            print(f"Starting worker {index}/{self.population_size}", end="\r", flush=True)
            worker.start()
        # queue checkpoints
        for id in range(self.population_size):
            print(f"Queuing checkpoint {id + 1}/{self.population_size}", end="\r", flush=True)
            # copy hyper-parameters
            hyper_parameters = copy.deepcopy(self.hyper_parameters)
            # create new checkpoint object
            checkpoint = Checkpoint(id, hyper_parameters)
            # prepare hyper-parameters
            self.evolver.prepare(hyper_parameters=checkpoint.hyper_parameters, logger=partial(self.log, checkpoint))
            # queue checkpoint for training
            self.train_queue.put(checkpoint)

    def write_to_tensorflow(self, checkpoint):
        """Plots loss and eval metric to tensorboard"""
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
        self.__tensorboard_writer.add_scalar(
            tag=f"Time/train/{checkpoint.id:03d}",
            scalar_value=checkpoint.train_time if checkpoint.train_time else 0,
            global_step=checkpoint.steps)
        self.__tensorboard_writer.add_scalar(
            tag=f"Time/eval/{checkpoint.id:03d}",
            scalar_value=checkpoint.eval_time if checkpoint.eval_time else 0,
            global_step=checkpoint.steps)
        self.__tensorboard_writer.add_scalar(
            tag=f"Time/evolve/{checkpoint.id:03d}",
            scalar_value=checkpoint.evolve_time if checkpoint.evolve_time else 0,
            global_step=checkpoint.steps)

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
                print(f"Queue length: {self.evolve_queue.qsize()}")
                # save checkpoint to database
                self.database.save_entry(checkpoint)
                # write to tensorboard if enabled
                if self.__tensorboard_writer:
                    self.write_to_tensorflow(checkpoint)
                # check if population is finished
                if self.is_population_finished(checkpoint):
                    self.log(checkpoint, "End criterium reached.")
                    self.end_event.set()
                    break
                # check if member is finished
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
                    start_evolve_time_ns = time.time_ns()
                    self.evolver.evolve(
                        member=checkpoint,
                        generation=self.database.get_latest,
                        population=self.database.to_list,
                        function=self.eval_function,
                        logger=partial(self.log, checkpoint))
                    checkpoint.evolve_time = float(time.time_ns() - start_evolve_time_ns) * float(10**(-9))
                # queue member for training
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