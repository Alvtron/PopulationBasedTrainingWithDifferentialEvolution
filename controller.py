import os
import math
import operator
import random
import pickle
import copy
import argparse
import time
import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
from functools import partial 
from checkpoint import Checkpoint
from member import Member
from utils.date import get_datetime_string
from utils.math import n_digits
from hyperparameters import Hyperparameters
from trainer import Trainer
from evaluator import Evaluator
from evolution import EvolveEngine
from database import SharedDatabase
from torch.utils.tensorboard import SummaryWriter

mp = torch.multiprocessing.get_context('spawn')

class Controller(object):
    def __init__(
            self, population_size : int, hyper_parameters : Hyperparameters,
            trainer : Trainer, evaluator : Evaluator, tester : Evaluator, evolver : EvolveEngine,
            loss_metric : str, eval_metric : str, loss_functions : dict,
            database : SharedDatabase, tensorboard_writer : SummaryWriter = None,
            step_size = 1, evolve_frequency : int = 5, end_criteria : dict = {'score': 100.0},
            detect_NaN : bool = False, device : str = 'cpu', verbose : bool = True, logging : bool = True):
        assert evolve_frequency and evolve_frequency > 0, f"Frequency must be of type {int} and 1 or higher."
        self.population_size = population_size
        self.hyper_parameters = hyper_parameters
        self.trainer = trainer
        self.evaluator = evaluator
        self.tester = tester
        self.evolver = evolver
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss_functions = loss_functions
        self.database = database
        self.step_size = step_size
        self.evolve_frequency = evolve_frequency
        self.end_criteria = end_criteria
        self.detect_NaN = detect_NaN
        self.device = device
        self.verbose = verbose
        self.logging = logging
        self.iterations = 0
        # create shared event for when to end training
        self.end_event = mp.Event()
        # create queues for training and evolving
        self.train_queue = mp.Queue(population_size)
        self.evolve_queue = mp.Queue(population_size)
        self.finish_queue = mp.Queue(population_size)
        self.__workers = []
        self.__tensorboard_writer = tensorboard_writer
        # TODO: implement save and load function for controller
        self.save_objective_info()

    def save_objective_info(self):
        parameters = {
            'population_size': self.population_size,
            'batch_size': self.trainer.batch_size,
            'device': self.device,
            'n_hyper_parameters': len(self.hyper_parameters),
            'hyper_parameters': self.hyper_parameters.parameter_paths(),
            'loss_metric': self.loss_metric,
            'eval_metric': self.eval_metric,
            'loss_functions': self.loss_functions,
            'step_size': self.step_size,
            'evolve_frequency': self.evolve_frequency,
            'end_criteria': self.end_criteria
        }
        pickle.dump(parameters, self.database.create_file("info", "parameters.obj").open("wb"))

    def __log(self, message : str):
        """Logs and prints the provided controller message in the appropriate syntax."""
        controller_name = self.__class__.__name__
        time = get_datetime_string()
        full_message = f"{time} {controller_name}, cycle {self.iterations:04d}: {message}"
        if self.logging:
            with self.database.create_file(tag='logs', file_name=f"{controller_name}_log.txt").open('a+') as file:
                file.write(full_message + '\n')
        if self.verbose:
            print(full_message)

    def __log_checkpoint(self, checkpoint : Checkpoint, message : str):
        """Logs and prints the provided checkpoint message in the appropriate syntax."""
        time = get_datetime_string()
        full_message = f"{time} {checkpoint}: {message}"
        if self.logging:
            with self.database.create_file(tag='logs', file_name=f"{checkpoint.id:03d}_log.txt").open('a+') as file:
                file.write(full_message + '\n')
        if self.verbose:
            print(full_message)
        if self.__tensorboard_writer:
            self.__tensorboard_writer.add_text(tag=f"member_{checkpoint.id:03d}", text_string=full_message, global_step=checkpoint.steps)

    def __write_to_tensorboard(self, checkpoint : Checkpoint):
        """Plots checkpoint data to tensorboard"""
        # plot eval metrics
        for eval_metric_group, eval_metrics in checkpoint.loss.items():
            for metric_name, metric_value in eval_metrics.items():
                self.__tensorboard_writer.add_scalar(
                    tag=f"metrics/{eval_metric_group}/{metric_name}/{checkpoint.id:03d}",
                    scalar_value=metric_value,
                    global_step=checkpoint.steps)
        # plot time
        for time_type, time_value in checkpoint.time.items():
            self.__tensorboard_writer.add_scalar(
                tag=f"time/{time_type}/{checkpoint.id:03d}",
                scalar_value=time_value,
                global_step=checkpoint.steps)
        # plot hyper-parameters
        for hparam_name, hparam in checkpoint.hyper_parameters:
            self.__tensorboard_writer.add_scalar(
                tag=f"Hyperparameters/{hparam_name}/{checkpoint.id:03d}",
                scalar_value=hparam.normalized,
                global_step=checkpoint.steps)

    def eval_function(self, checkpoint):
        current_model_state, _, _, _ , _ = self.trainer.train(
            hyper_parameters=checkpoint.hyper_parameters,
            model_state=checkpoint.model_state,
            optimizer_state=checkpoint.optimizer_state,
            epochs=checkpoint.epochs,
            steps=checkpoint.steps,
            step_size=1
            )
        loss = self.evaluator.eval(current_model_state)
        return loss[self.eval_metric]

    def is_member_ready(self, checkpoint : Checkpoint):
        """True every n-th epoch."""
        return checkpoint.steps % self.evolve_frequency == 0

    def is_member_finished(self, checkpoint : Checkpoint):
        if 'epochs' in self.end_criteria and checkpoint.epochs >= self.end_criteria['epochs']:
            # the number of epochs is equal or above the given treshold
            return True
        if 'steps' in self.end_criteria and checkpoint.steps >= self.end_criteria['steps']:
            # the number of steps is equal or above the given treshold
            return True
        return False
    
    def is_population_finished(self, checkpoint : Checkpoint):
        if 'score' in self.end_criteria and checkpoint.score >= self.end_criteria['score']:
            # score is above the given treshold
            return True
        return False

    def spawn_workers(self):
        self.__workers = [
            Member(
                end_event = self.end_event,
                trainer = self.trainer,
                evaluator = self.evaluator,
                evolve_queue = self.evolve_queue,
                train_queue = self.train_queue)
            for _ in range(self.population_size)]
        # Starting workers
        for index, worker in enumerate(self.__workers, start=1):
            #print(f"Starting worker {index}/{self.population_size}", end="\r", flush=True)
            self.__log(f"Starting worker {index}/{self.population_size}")
            worker.start()

    def queue_members(self):
        for id in range(self.population_size):
            # copy hyper-parameters
            hyper_parameters = copy.deepcopy(self.hyper_parameters)
            # create new checkpoint object
            checkpoint = Checkpoint(
                id=id,
                hyper_parameters=hyper_parameters,
                loss_metric=self.loss_metric,
                eval_metric=self.eval_metric,
                step_size=1)
            # prepare hyper-parameters
            self.evolver.prepare(hyper_parameters=checkpoint.hyper_parameters, logger=partial(self.__log_checkpoint, checkpoint))
            # queue checkpoint for training
            self.__log_checkpoint(checkpoint, "queued for training...")
            self.train_queue.put(checkpoint)

    def start_training_procedure(self):
        self.__log("Creating worker processes...")
        self.spawn_workers()
        self.__log("Queing member checkpoints...")
        self.queue_members()

    def start(self):
        self.iterations = 0
        try:
            # start training
            self.start_training_procedure()
            # controller loop
            while not self.end_event.is_set():
                if self.evolve_queue.empty():
                    continue
                # get next checkpoint
                checkpoint = self.evolve_queue.get()
                queue_length = self.evolve_queue.qsize()
                if queue_length > 0:
                    self.__log(f"queue size: {queue_length}")
                # check for nan loss-value
                if self.detect_NaN and any(math.isnan(value) for value in checkpoint.loss['eval'].values()):
                    self.__log_checkpoint(checkpoint, "NaN metric detected.")
                    self.__log_checkpoint(checkpoint, "stopping.")
                    self.evolver.population_size -=1
                    self.database.remove_latest(checkpoint.id)
                    self.finish_queue.put(checkpoint)
                    if self.finish_queue.full():
                        self.__log("All workers are finished.")
                        self.end_event.set()
                        break
                    else: continue
                # save checkpoint to database
                self.database.update(checkpoint.id, checkpoint.steps, checkpoint)
                # write to tensorboard if enabled
                if self.__tensorboard_writer:
                    self.__write_to_tensorboard(checkpoint)
                # check if population is finished
                if self.is_population_finished(checkpoint):
                    self.__log_checkpoint(checkpoint, "End criterium reached.")
                    self.end_event.set()
                    break
                # check if member is finished
                if self.is_member_finished(checkpoint):
                    self.__log_checkpoint(checkpoint, "finished.")
                    self.finish_queue.put(checkpoint)
                    if self.finish_queue.full():
                        self.__log("All workers are finished.")
                        self.end_event.set()
                        break
                    else: continue
                # evolve member if ready
                if self.is_member_ready(checkpoint):
                    self.__log_checkpoint(checkpoint, "evolving...")
                    start_evolve_time_ns = time.time_ns()
                    self.evolver.evolve(
                        member=checkpoint,
                        population=self.database.latest(),
                        function=self.eval_function,
                        logger=partial(self.__log_checkpoint, checkpoint))
                    checkpoint.time['evolve'] = float(time.time_ns() - start_evolve_time_ns) * float(10**(-9))
                # queue member for training
                self.__log_checkpoint(checkpoint, "training...")
                self.train_queue.put(checkpoint)
                self.iterations += 1
            # terminate worker processes
            self.__log("Controller is finished.")
        except KeyboardInterrupt:
            self.__log("Controller was interupted.")
        finally:
            self.__log("Terminating all left-over worker-processes...")
            for worker in self.__workers:
                if isinstance(worker, mp.Process):
                    worker.terminate()
                else: continue
            self.__log("Termination was successfull!")
