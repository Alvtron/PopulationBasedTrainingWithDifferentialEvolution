import os
import math
import random
import pickle
import copy
import time
import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
from typing import List, Dict
from functools import partial 
from member import Checkpoint
from worker import Worker
from utils.date import get_datetime_string
from hyperparameters import Hyperparameters
from trainer import Trainer
from evaluator import Evaluator
from evolution import EvolveEngine
from database import Database
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from torch.multiprocessing import Process
from multiprocessing.context import BaseContext

STOP_FLAG = None

class Controller(object):
    def __init__(
            self, context : BaseContext, hyper_parameters : Hyperparameters,
            trainer : Trainer, evaluator : Evaluator, tester : Evaluator, evolver : EvolveEngine,
            loss_metric : str, eval_metric : str, loss_functions : dict, database : Database,
            step_size = 1, evolve_frequency : int = 5, end_criteria : dict = {'score': 100.0}, eval_steps=1,
            tensorboard_writer : SummaryWriter = None, detect_NaN : bool = False, history_limit : int = None,
            device : str = 'cpu', n_jobs : int = -1, verbose : int = 1, logging : bool = True):
        assert evolve_frequency and evolve_frequency > 0, f"Frequency must be of type {int} and 1 or higher."
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
        self.eval_steps = eval_steps
        self.detect_NaN = detect_NaN
        self.device = device
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.logging = logging
        self.generations = 0
        self.context = context
        # create shared event for when to end training
        self.end_event = self.context.Event()
        # create queues for training and evolving
        self.train_queue = self.context.Queue()
        self.evolve_queue = self.context.Queue()
        self.finish_queue = self.context.Queue()
        self.__workers = []
        self.__n_active_workers = 0
        self.history_limit = history_limit
        self.scoreboard = dict()
        self.__tensorboard_writer = tensorboard_writer
        # TODO: implement save and load function for controller
        self.save_objective_info()

    def __print(self, message : str):
        """Prints the provided controller message in the appropriate syntax."""
        controller_name = self.__class__.__name__
        time = get_datetime_string()
        full_message = f"{time} (G{self.generations:03d}) {controller_name}: {message}"
        print(full_message)

    def __say(self, message : str):
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 0."""
        if self.verbose > 0: self.__print(message)

    def __whisper(self, message : str):
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 1."""
        if self.verbose > 1: self.__print(message)

    def __log(self, member : Checkpoint, message : str, verbose=True):
        """Logs and prints the provided member message in the appropriate syntax."""
        time = get_datetime_string()
        full_message = f"{time} (G{self.generations:03d}) {member}: {message}"
        if self.logging:
            with self.database.create_file(tag='logs', file_name=f"{member.id:03d}_log.txt").open('a+') as file:
                file.write(full_message + '\n')
        if verbose and self.verbose > 0:
            print(full_message)
        if self.__tensorboard_writer:
            self.__tensorboard_writer.add_text(tag=f"member_{member.id:03d}", text_string=full_message, global_step=member.steps)

    def __log_silent(self, member : Checkpoint, message : str):
        self.__log(member, message, verbose=self.verbose > 1)

    def update_population(self, member : Checkpoint):
        """
        Saves the provided member to the population.
        In addition, the method saves the provided member to a file on id/steps in the database directory.
        """
        # Save entry to population (in memory). This replaces the old entry.
        self.evolver.population[member.id] = member
        # Save entry to database directory.
        self.database.update(member.id, member.steps, member)
        # save score to history
        self.scoreboard[(member.id, member.steps)] = member.score()

    def garbage_collection(self):
        if not self.history_limit or not self.scoreboard:
            return
        excess = len(self.scoreboard) - self.history_limit
        if excess < 1:
            return
        minimize = self.loss_functions[self.eval_metric].minimize
        worst_entries = sorted(self.scoreboard.items(), key=lambda x: x[1], reverse=minimize)
        excess_entries = worst_entries[:excess]
        for (id, steps), _ in excess_entries:
            self.__whisper(f"deleting the model and optimizer state from member {id} at step {steps} ...")
            del self.scoreboard[(id, steps)]
            entry = self.database.entry(id, steps)
            entry.model_state = None
            entry.optimizer_state = None
            self.database.update(id, steps, entry)

    def is_removed(self, member: Checkpoint):
        """Finds out if the member in question is removed from the population"""
        return self.generations > 0 and member.id not in self.evolver.population

    def stop_member(self, member : Checkpoint):
        self.finish_queue.put(member)
        self.adjust_workers_size()
        
    def adjust_workers_size(self):
        delta_size = self.__n_active_workers - self.evolver.population_size
        # adjust down if number of active workers is higher than the population size
        if delta_size > 0:
            self.stop_workers(k=delta_size)
        # adjust up if number of active workers is lower than the population size
        # and if adjustment does not exceed n_jobs
        elif delta_size < 0:
            new_size = min(abs(delta_size) + self.__n_active_workers, self.n_jobs)
            self.spawn_workers(k=new_size - self.__n_active_workers)

    def stop_workers(self, k : int = None):
        if not k:
            k = len(self.__workers)
        elif isinstance(k, int) and 0 < k < len(self.__workers):
            k = k
        else:
            raise IndexError()
        for _ in range(k):
            self.train_queue.put(STOP_FLAG)
        self.__n_active_workers -= k

    def create_workers(self, k):
        return [
            Worker(
                id=id,
                end_event_global=self.end_event,
                trainer=self.trainer,
                evaluator=self.evaluator,
                evolve_queue=self.evolve_queue,
                train_queue=self.train_queue,
                random_seed=id,
                verbose=self.verbose > 1)
            for id in range(k)]

    def save_objective_info(self):
        parameters = {
            'population_size': self.evolver.population_size,
            'batch_size': self.trainer.batch_size,
            'device': self.device,
            'hyper_parameters': self.hyper_parameters.keys(),
            'loss_metric': self.loss_metric,
            'eval_metric': self.eval_metric,
            'loss_functions': self.loss_functions,
            'step_size': self.step_size,
            'evolve_frequency': self.evolve_frequency,
            'end_criteria': self.end_criteria
        }
        pickle.dump(parameters, self.database.create_file("info", "parameters.obj").open("wb"))


    def __write_to_tensorboard(self, member : Checkpoint):
        """Plots member data to tensorboard"""
        # plot eval metrics
        for eval_metric_group, eval_metrics in member.loss.items():
            for metric_name, metric_value in eval_metrics.items():
                self.__tensorboard_writer.add_scalar(
                    tag=f"metrics/{eval_metric_group}/{metric_name}/{member.id:03d}",
                    scalar_value=metric_value,
                    global_step=member.steps)
        # plot time
        for time_type, time_value in member.time.items():
            self.__tensorboard_writer.add_scalar(
                tag=f"time/{time_type}/{member.id:03d}",
                scalar_value=time_value,
                global_step=member.steps)
        # plot hyper-parameters
        for hparam_name, hparam in member.hyper_parameters:
            self.__tensorboard_writer.add_scalar(
                tag=f"Hyperparameters/{hparam_name}/{member.id:03d}",
                scalar_value=hparam.normalized if hparam.is_categorical else hparam.value,
                global_step=member.steps)

    def is_member_ready(self, member : Checkpoint):
        """True every n-th epoch."""
        return member.steps % self.evolve_frequency == 0

    def is_member_finished(self, member : Checkpoint):
        """With the end_criteria, check if the provided member is finished training."""
        if 'epochs' in self.end_criteria and member.epochs >= self.end_criteria['epochs']:
            # the number of epochs is equal or above the given treshold
            return True
        if 'steps' in self.end_criteria and member.steps >= self.end_criteria['steps']:
            # the number of steps is equal or above the given treshold
            return True
        return False
    
    def is_population_finished(self, member : Checkpoint):
        """
        With the end_criteria, check if the entire population is finished
        by inspecting the provided member.
        """
        if 'score' in self.end_criteria or self.has_NaN_value(member):
            return False
        return member.score() >= self.end_criteria['score']

    def spawn_workers(self, k):
        """Spawn a worker process for every member in the population."""
        new_workers = self.create_workers(k)
        # Starting workers
        for index, worker in enumerate(new_workers, start=1):
            self.__whisper(f"starting worker {index}/{len(new_workers)}")
            worker.start()
        self.__n_active_workers += k
        self.__workers += new_workers

    def create_member(self, id):
        """Create a member object"""
        # copy hyper-parameters
        hyper_parameters = copy.deepcopy(self.hyper_parameters)
        # create new member object
        member = Checkpoint(
            id=id,
            hyper_parameters=hyper_parameters,
            loss_metric=self.loss_metric,
            eval_metric=self.eval_metric,
            minimize=self.loss_functions[self.eval_metric].minimize,
            step_size=self.step_size)
        return member

    def create_members(self, k : int):
        members = dict()
        for id in range(k):
            members[id] = self.create_member(id)
        return members

    def queue_members(self, members : List[Checkpoint]):
        """Queue members for training."""
        for index, member in enumerate(members, start=1):
            self.train_queue.put(member)
            self.__whisper(f"queued member {index}/{len(members)} for training...")

    def has_NaN_value(self, member : Checkpoint):
        """Returns True if the member has NaN values for its loss_metric or eval_metric."""
        for loss_type, loss_value in member.loss['eval'].items():
            if loss_type in (self.loss_metric, self.eval_metric) and math.isnan(loss_value):
                return True
        return False

    def on_NaN_detect(self, member : Checkpoint):
        """
        Handle member with NaN loss. Depending on the active population size,
        generate a new member start-object or sample one from the population.
        """
        self.__log_silent(member, "NaN metric detected.")
        population = list(c for c in self.evolver.population.values() if member.id != c.id)
        if 0 == len(population) or len(population) < self.evolver.population_size - 1:
            self.__log_silent(member, "population is not full. Creating new member.")
            member = self.create_member(member.id)
        else:
            random_member = random.choice(population)
            self.__log_silent(member, f"replacing with existing member {random_member.id}")
            member.update(random_member)

    def eval_function(self, trial : Checkpoint):
        """
        Evaluates the provided trial checkpoint.
        Returns the trial checkpoint.
        The checkpoint object holds the loss result.
        """
        trial.model_state, trial.optimizer_state, trial.epochs, trial.steps, trial.loss['train'] = self.trainer.train(
            hyper_parameters=trial.hyper_parameters,
            model_state=trial.model_state,
            optimizer_state=trial.optimizer_state,
            epochs=trial.epochs,
            steps=trial.steps,
            step_size=self.eval_steps
            )
        trial.loss['eval'] = self.evaluator.eval(trial.model_state)
        return trial

    def start_training_procedure(self):
        """
        Creates worker-processes for training and evaluation.
        Creates, prepares and queue member-objects for training.
        """
        self.__whisper("creating initial members member objects...")
        members = self.create_members(self.evolver.population_size)
        self.__whisper("on evolver initialization")
        self.evolver.on_initialization(members, self.__log_silent)
        self.__whisper("creating worker processes...")
        n_workers = self.n_jobs if 0 < self.n_jobs <= self.evolver.population_size else self.evolver.population_size
        self.spawn_workers(n_workers)
        self.__whisper("queuing member members...")
        self.queue_members(members.values())

    def start(self, synchronized : bool = True):
        """
        Start global training procedure. Ends when end_criteria is met.
        """
        try:
            # start controller loop
            self.__say("Starting training procedure...")
            self.train_synchronously() if synchronized else self.train_asynchronous()
            # terminate worker processes
            self.__say("controller is finished.")
        except KeyboardInterrupt:
            self.__say("controller was interupted.")
        finally:
            self.__whisper("sending stop signal to the remaining worker processes...")
            for worker in self.__workers:
                self.train_queue.put(STOP_FLAG)
            self.__whisper("waiting for all worker processes to finish...")
            for worker in self.__workers:
                worker.join()
            self.__whisper("termination was successfull!")

    def train_asynchronous(self):
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        and the controller does not mind the order of the members.\n

        Generations are counted each n-th member queued for training. 
        """
        self.generations = 0
        self.start_training_procedure()
        while not self.end_event.is_set():
            # prepare generation with evolver
            self.__whisper("on generation start")
            self.evolver.on_generation_start(self.__log_silent)
            iterations = 0
            while iterations < self.evolver.population_size:
                # get next member
                self.__whisper("awaiting next trained member...")
                member = self.evolve_queue.get()
                if member is STOP_FLAG:
                    return
                # check if member is removed from population
                if self.is_removed(member):
                    self.__log_silent(member, "removed from the population.")
                    self.stop_member(member)
                    continue
                self.__whisper(f"queue size: {self.evolve_queue.qsize()}")
                # update member in population/database
                self.__log_silent(member, "updating in population/database...")
                self.update_population(member)
                # process member
                self.process_member(member)
                iterations += 1
            # process generation with evolver
            self.__whisper("on generation end")
            self.evolver.on_generation_end(self.__log_silent)
            # perform garbage collection
            self.__whisper("performing garbage collection...")
            self.garbage_collection()
            self.generations += 1

    def train_synchronously(self):
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        but they are waiting for each other between each generation cycle.
        """
        self.generations = 0
        self.start_training_procedure()
        while not self.end_event.is_set():
            retrieved_members = 0
            while retrieved_members < self.evolver.population_size:
                # get next member
                self.__whisper("awaiting next trained member...")
                member = self.evolve_queue.get()
                if member is STOP_FLAG:
                    return
                # check if member is removed from population
                if self.is_removed(member):
                    self.__log_silent(member, "removed from the population.")
                    self.stop_member(member)
                    continue
                # update member in population/database
                self.__log_silent(member, "updating in population/database...")
                self.update_population(member)
                retrieved_members += 1
            #update population size:
            self.__whisper("on generation start")
            self.evolver.on_generation_start(self.__log_silent)
            for member in self.evolver.population.values():
                self.process_member(member)
            self.__whisper("on generation end")
            self.evolver.on_generation_end(self.__log_silent)
            # perform garbage collection
            self.__whisper("performing garbage collection...")
            self.garbage_collection()
            self.generations += 1

    def check_if_finished(self, member : Checkpoint):
        # check if population is finished
        if self.is_population_finished(member):
            self.__say(f"member {member.id} reached the end criterium.")
            self.finish_queue.put(member)
            self.evolver.population_size -= 1
            self.stop()
            return True
        # check if member is finished
        if self.is_member_finished(member):
            self.__log(member, "finished.")
            self.finish_queue.put(member)
            self.evolver.population_size -= 1
            if self.evolver.population_size == 0:
                self.__say("population finished. All members have reached the end-criteria.")
                self.stop()
            return True
        return False

    def stop(self):
        self.end_event.set()
        self.evolve_queue.put(STOP_FLAG)

    def process_member(self, member : Checkpoint):
        """The provided member checkpoint is analyzed, validated, changed (if valid) and queued for training (if valid)."""
        # log performance
        self.__log(member, member.performance_details())
        # write to tensorboard if enabled
        if self.__tensorboard_writer:
            self.__write_to_tensorboard(member)
        if self.check_if_finished(member):
            return
        # make a copy of the member
        candidate = member.copy()
        # check for nan loss if enabled
        if self.detect_NaN and self.has_NaN_value(candidate):
            self.on_NaN_detect(candidate)
            self.train_member(candidate)
            return
        # check population size
        if len(self.evolver.population) < self.evolver.population_size:
            self.__log_silent(candidate, f"population is too small ({len(self.evolver.population)} < {self.evolver.population_size}). Skipping.")
            self.train_member(candidate)
            return
        # evolve member if ready
        if self.is_member_ready(candidate):
            start_evolve_time_ns = time.time_ns()
            self.__log_silent(candidate, "evolving...")
            trial = candidate.copy()
            trial = self.evolver.on_evolve(
                member=trial,
                logger=partial(self.__log_silent, candidate))
            self.__log_silent(candidate, "evaluating mutant...")
            trial = self.evolver.on_evaluation(
                member=candidate,
                trial=trial,
                eval_function=self.eval_function,
                logger=partial(self.__log_silent, candidate))
            self.__log_silent(candidate, "updating...")
            candidate.update(trial)
            candidate.time['evolve'] = float(time.time_ns() - start_evolve_time_ns) * float(10**(-9))
        # train candidate
        self.train_member(candidate)

    def train_member(self, member : Checkpoint):
        """Queue the provided member checkpoint for training."""
        self.__log_silent(member, "training...")
        self.train_queue.put(member)
