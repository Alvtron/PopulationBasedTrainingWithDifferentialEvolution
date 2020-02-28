import os
import sys
import math
import time
import random
import copy
import pickle
import shutil
import warnings
from typing import List, Dict, Sequence
from functools import partial 
from collections import defaultdict
from multiprocessing.context import BaseContext
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

import torch
import torchvision
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Process
import matplotlib.pyplot as plt

import pbt.member
from .trainingservice import TrainingService
from .member import Checkpoint, Population, Generation, clear_member_states
from .utils.date import get_datetime_string
from .hyperparameters import DiscreteHyperparameter, Hyperparameters
from .trainer import Trainer
from .evaluator import Evaluator
from .evolution import EvolveEngine
from .database import Database

class Controller(object):
    def __init__(
            self, context : BaseContext, population_size : int, hyper_parameters : Hyperparameters,
            trainer : Trainer, evaluator : Evaluator, evolver : EvolveEngine,
            loss_metric : str, eval_metric : str, loss_functions : dict, database : Database,
            step_size = 1, end_criteria : dict = {'score': 100.0},
            tensorboard_writer : SummaryWriter = None, detect_NaN : bool = False, history_limit : int = None,
            devices : List[str] = ['cpu'], n_jobs : int = -1, threading : bool = False, verbose : int = 1, logging : bool = True):
        assert step_size and step_size > 0, f"Step size must be of type {int} and 1 or higher."
        self.population = Population(population_size)
        self.database = database
        self.evolver = evolver
        self.hyper_parameters = hyper_parameters
        self.training_service = TrainingService(context=context, trainer=trainer, evaluator=evaluator,
            devices=devices, n_jobs=n_jobs, threading=threading, verbose=verbose)
        self.step_size = step_size
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss_functions = loss_functions
        self.end_criteria = end_criteria
        self.detect_NaN = detect_NaN
        self.verbose = verbose
        self.logging = logging
        self.history_limit = history_limit if history_limit and history_limit > 2 else 2
        self.__tensorboard_writer = tensorboard_writer
        self.__nfe = 0

    def __str__(self):
        controller = self.__class__.__name__
        time = get_datetime_string()
        generation = f"G{len(self.population.generations):03d}"
        nfe = f"({self.__nfe}/{self.end_criteria['nfe']})" if 'nfe' in self.end_criteria and self.end_criteria['nfe'] else self.__nfe
        return f"{time} {nfe} {generation} {controller}"

    def __print(self, message : str):
        """Prints the provided controller message in the appropriate syntax."""
        print(f"{self}: {message}")

    def __say(self, message : str):
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 0."""
        if self.verbose > 0: self.__print(message)

    def __whisper(self, message : str):
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 1."""
        if self.verbose > 1: self.__print(message)

    def __log(self, member : Checkpoint, message : str, verbose=True):
        """Logs and prints the provided member message in the appropriate syntax."""
        time = get_datetime_string()
        nfe = f"({self.__nfe}/{self.end_criteria['nfe']})" if 'nfe' in self.end_criteria and self.end_criteria['nfe'] else self.__nfe
        full_message = f"{time} {nfe} G{len(self.population.generations):03d} {member}: {message}"
        if self.logging:
            with self.database.create_file(tag='logs', file_name=f"{member.id:03d}_log.txt").open('a+') as file:
                file.write(full_message + '\n')
        if verbose and self.verbose > 0:
            print(full_message)
        if self.__tensorboard_writer:
            self.__tensorboard_writer.add_text(tag=f"member_{member.id:03d}", text_string=full_message, global_step=member.steps)

    def __log_silent(self, member : Checkpoint, message : str):
        self.__log(member, message, verbose=self.verbose > 1)

    def __update_tensorboard(self):
        """Plots member data to tensorboard"""
        for member in self.population:
            # plot eval metrics
            for eval_metric_group, eval_metrics in member.loss.items():
                for metric_name, metric_value in eval_metrics.items():
                    self.__tensorboard_writer.add_scalar(
                        tag=f"metrics/{eval_metric_group}_{metric_name}/{member.id:03d}",
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
                    tag=f"hyperparameters/{hparam_name}/{member.id:03d}",
                    scalar_value=hparam.normalized if isinstance(hparam, DiscreteHyperparameter) else hparam.value,
                    global_step=member.steps)

    def create_member(self, id):
        """Create a member object"""
        # create new member object
        member = Checkpoint(
            id=id,
            hyper_parameters=copy.deepcopy(self.hyper_parameters),
            loss_metric=self.loss_metric,
            eval_metric=self.eval_metric,
            minimize=self.loss_functions[self.eval_metric].minimize)
        # let evolver process new member
        self.evolver.on_member_spawn(member, self.__whisper)
        return member

    def create_members(self, k : int):
        members = list()
        for id in range(k):
            members.append(self.create_member(id))
        return members

    def remove_bad_member_states(self):
        if self.history_limit is None:
            return
        if len(self.population.generations) < self.history_limit + 1:
            return
        for member in self.population.generations[-(self.history_limit + 1)]:
            if not member.has_state():
                # skipping the member as it has no state to delete
                continue
            self.__whisper(f"deleting the state from member {member.id} at step {member.steps} with score {member.score():.4f}...")
            member.delete_state()
            # updating database
            self.database.update(member.id, member.steps, member)
        # clean tmp states
        for state_folder in pbt.member.MEMBER_STATE_DIRECTORY.glob("*"):
            step = int(state_folder.name)
            if step < len(self.population.generations) * self.step_size - self.history_limit * self.step_size: 
                shutil.rmtree(state_folder)

    def update_database(self, member : Checkpoint):
        """Updates the database stored in files."""
        self.__whisper(f"updating member {member.id} in database...")
        member.load_state(device='cpu')
        self.database.update(member.id, member.steps, member)
        member.unload_state()

    def is_member_finished(self, member : Checkpoint):
        """With the end_criteria, check if the provided member is finished training."""
        if 'steps' in self.end_criteria and self.end_criteria['steps'] and member.steps >= self.end_criteria['steps']:
            # the number of steps is equal or above the given treshold
            return True
        return False
    
    def is_population_finished(self):
        """
        With the end_criteria, check if the entire population is finished
        by inspecting the provided member.
        """
        if 'nfe' in self.end_criteria and self.end_criteria['nfe'] and self.__nfe >= self.end_criteria['nfe']:
            return True
        if 'score' in self.end_criteria and self.end_criteria['score'] and any(not self.has_NaN_value(member) and member >= self.end_criteria['score'] for member in self.population):
            return True
        if all(self.is_member_finished(member) for member in self.population):
            return True
        return False

    def has_NaN_value(self, member : Checkpoint):
        """Returns True if the member has NaN values for its loss_metric or eval_metric."""
        for loss_type, loss_value in member.loss['eval'].items():
            if loss_type in (self.loss_metric, self.eval_metric) and math.isnan(loss_value):
                return True
        return False

    def on_NaN_detect(self, member : Checkpoint):
        """
        Handle member with NaN loss. Depending on the active population size,
        generate a new member start-object or sample one from the previous generation.
        """
        # create new member if population is not fully initialized yet
        if len(self.population.generations) < 2:
            self.replace_member_with_new(member)
        else:
            self.replace_member_with_best(member)

    def replace_member_with_new(self, member : Checkpoint):
        """Replace member with a new member."""
        self.__log_silent(member, "creating new member...")
        new_member = self.create_member(member.id)
        member.delete_state()
        member.update(new_member)
        member.steps = new_member.steps
        member.epochs = new_member.epochs

    def replace_member_with_best(self, member : Checkpoint, p = 0.2):
        """Replace member with one of the best performing members in a previous generation."""
        previous_generation = self.population.generations[-2]
        n_elitists = max(1, round(len(previous_generation) * p))
        elitists = sorted((m for m in previous_generation if m != member), reverse=True)[:n_elitists]
        elitist = random.choice(elitists)
        self.__log_silent(member, f"replacing with member {elitist.id}...")
        member.update(elitist)
        # resample hyper parameters
        [hp.sample_uniform() for hp in member]

    def initialize_population(self):
        new_members = self.create_members(k=self.population.size)
        for member in self.training_service.train(new_members, self.step_size):
            # log performance
            self.__log(member, member.performance_details())
            # Add member to population.
            self.__whisper(f"updating member {member.id} in population...")
            self.population.append(member)
            # Save member to database directory.
            self.update_database(member)

    def start(self, use_old = False):
        """
        Start global training procedure. Ends when end_criteria is met.
        """
        # reset member state folder
        clear_member_states()
        try:
            # start controller loop
            self.__say("Starting training procedure...")
            if not use_old:
                self.train_synchronously()
            else:
                self.train_synchronously_old()
            # terminate worker processes
            self.__say("controller is finished.")
        except KeyboardInterrupt:
            self.__say("controller was interupted.")
            # close pool
            self.training_service.stop()
        # reset member state folder
        clear_member_states()

    def train_synchronously(self):
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        but they are waiting for each other between each generation cycle.
        """
        self.training_service.start()
        self.initialize_population()
        while not self.is_population_finished():
            self.__whisper("on generation start")
            self.evolver.on_generation_start(self.population, self.__whisper)
            # generate new candidates
            new_candidates = [self.evolver.on_evolve(member, self.population, partial(self.__log_silent, member)) for member in self.population]
            self.population.new_generation()
            for candidates in self.training_service.train(new_candidates, self.step_size):
                member = self.evolver.on_evaluation(candidates, self.__whisper)
                self.__nfe += 1 #if isinstance(candidates, Checkpoint) else len(candidates)
                # log performance
                self.__log(member, member.performance_details())
                # Add member to population.
                self.__whisper(f"updating member {member.id} in population...")
                self.population.append(member)
                # Save member to database directory.
                self.update_database(member)
            self.__whisper("on generation end")
            self.evolver.on_generation_end(self.population, self.__whisper)
            # write to tensorboard if enabled
            if self.__tensorboard_writer:
                self.__update_tensorboard()
            # perform garbage collection
            self.__whisper("performing garbage collection...")
            self.remove_bad_member_states()
        self.__say(f"end criteria has been reached.")
        self.training_service.stop()

    def train_synchronously_old(self, eval_steps=1):
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        but they are waiting for each other between each generation cycle.
        """
        self.training_service.start()
        self.initialize_population()
        while not self.is_population_finished():
            self.__whisper("on generation start")
            self.evolver.on_generation_start(self.population, self.__whisper)
            # generate new candidates
            new_candidates = [self.evolver.on_evolve(member, self.population, partial(self.__log_silent, member)) for member in self.population]
            self.population.new_generation()
            if isinstance(next(iter(new_candidates)), Checkpoint):
                for candidates in self.training_service.train(new_candidates, self.step_size):
                    member = self.evolver.on_evaluation(candidates, self.__whisper)
                    self.__nfe += 1 #if isinstance(candidates, Checkpoint) else len(candidates)
                    # log performance
                    self.__log(member, member.performance_details())
                    # Add member to population.
                    self.__whisper(f"updating member {member.id} in population...")
                    self.population.append(member)
                    # Save member to database directory.
                    self.update_database(member)
            else:
                # eval candidates
                best_candidates = list()
                for candidates in self.training_service.train(new_candidates, eval_steps):
                    member = self.evolver.on_evaluation(candidates, self.__whisper)
                    best_candidates.append(member)
                    self.__nfe += 1 #if isinstance(candidates, Checkpoint) else len(candidates)
                # train best
                for member in self.training_service.train(best_candidates, self.step_size - eval_steps):
                    # log performance
                    self.__log(member, member.performance_details())
                    # Add member to population.
                    self.__whisper(f"updating member {member.id} in population...")
                    self.population.append(member)
                    # Save member to database directory.
                    self.update_database(member)
            self.__whisper("on generation end")
            self.evolver.on_generation_end(self.population, self.__whisper)
            # write to tensorboard if enabled
            if self.__tensorboard_writer:
                self.__update_tensorboard()
            # perform garbage collection
            self.__whisper("performing garbage collection...")
            self.remove_bad_member_states()
        self.__say(f"end criteria has been reached.")
        # close pool
        self.training_service.stop()