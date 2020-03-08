import os
import sys
import math
import time
import random
import copy
import pickle
import shutil
import warnings
from typing import List, Dict, Sequence, Iterator
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
            self, population_size : int, hyper_parameters : Hyperparameters,
            trainer : Trainer, evaluator : Evaluator, evolver : EvolveEngine,
            loss_metric : str, eval_metric : str, loss_functions : dict, database : Database,
            step_size = 1, end_criteria : dict = {'score': 100.0},
            tensorboard_writer : SummaryWriter = None, detect_NaN : bool = False, history_limit : int = None,
            devices : List[str] = ['cpu'], n_jobs : int = -1, threading : bool = False, verbose : int = 1, logging : bool = True):
        assert step_size and step_size > 0, f"Step size must be of type {int} and 1 or higher."
        self.population_size = population_size
        self.population = Population()
        self.database = database
        self.evolver = evolver
        self.hyper_parameters = hyper_parameters
        self.training_service = TrainingService(trainer=trainer, evaluator=evaluator,
            devices=devices, n_jobs=n_jobs, threading=threading, verbose=verbose>2)
        self.step_size = step_size
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss_functions = loss_functions
        self.end_criteria = end_criteria
        self.detect_NaN = detect_NaN
        self.verbose = verbose
        self.logging = logging
        self.history_limit = history_limit if history_limit and history_limit > 2 else 2
        self.nfe = 0
        self._tensorboard_writer = tensorboard_writer

    def __create_message(self, message : str, tag : str = None):
        time = get_datetime_string()
        generation = f"G{len(self.population.generations):03d}"
        nfe = f"({self.nfe}/{self.end_criteria['nfe']})" if 'nfe' in self.end_criteria and self.end_criteria['nfe'] else self.nfe
        return f"{time} {nfe} {generation}{f' {tag}' if tag else ''}: {message}"

    def _say(self, message : str, member : Checkpoint = None):
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 0."""
        self.__register(message=message, verbosity=0, member=member)

    def _whisper(self, message : str, member = None):
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 1."""
        self.__register(message=message, verbosity=1, member=member)

    def __register(self, message : str, verbosity : int, member : Checkpoint = None):
        """Logs and prints the provided message in the appropriate syntax. If a member is provided, the message is attached to that member."""
        print_tag = member if member else self.__class__.__name__
        file_tag = f"member_{member.id}" if member else self.__class__.__name__
        full_message = self.__create_message(message, tag=print_tag)
        self.__print(message=full_message, verbosity=verbosity)
        self.__log_to_file(message=full_message, tag=file_tag)
        self.__log_to_tensorboard(message=full_message, tag=file_tag, global_steps=member.steps if member else None)
    
    def __print(self, message : str, verbosity : int):
        if self.verbose > verbosity:
            print(message)

    def __log_to_file(self, message : str, tag : str):
        if not self.logging:
            return
        with self.database.create_file(tag='logs', file_name=f"{tag}_log.txt").open('a+') as file:
            file.write(message + '\n')
    
    def __log_to_tensorboard(self, message : str, tag : str, global_steps : int = None):
        if not self._tensorboard_writer:
            return
        self._tensorboard_writer.add_text(tag=tag, text_string=message, global_step=global_steps)

    def _update_tensorboard(self, member : Checkpoint):
        """Plots member data to tensorboard"""
        if not self._tensorboard_writer:
            return       
        # plot eval metrics
        for eval_metric_group, eval_metrics in member.loss.items():
            for metric_name, metric_value in eval_metrics.items():
                self._tensorboard_writer.add_scalar(
                    tag=f"metrics/{eval_metric_group}_{metric_name}/{member.id:03d}",
                    scalar_value=metric_value,
                    global_step=member.steps)
        # plot time
        for time_type, time_value in member.time.items():
            self._tensorboard_writer.add_scalar(
                tag=f"time/{time_type}/{member.id:03d}",
                scalar_value=time_value,
                global_step=member.steps)
        # plot hyper-parameters
        for hparam_name, hparam in member.parameters:
            self._tensorboard_writer.add_scalar(
                tag=f"hyperparameters/{hparam_name}/{member.id:03d}",
                scalar_value=hparam.normalized if isinstance(hparam, DiscreteHyperparameter) else hparam.value,
                global_step=member.steps)

    def create_member(self, id):
        """Create a member object"""
        # create new member object
        member = Checkpoint(
            id=id,
            parameters=copy.deepcopy(self.hyper_parameters),
            loss_metric=self.loss_metric,
            eval_metric=self.eval_metric,
            minimize=self.loss_functions[self.eval_metric].minimize)
        # process new member with evolver
        self.evolver.on_member_spawn(member, self._whisper)
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
            if not member.has_state_files():
                # skipping the member as it has no state to delete
                continue
            self._whisper(f"deleting the state from member {member.id} at step {member.steps} with score {member.score():.4f}...")
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
        self._whisper(f"updating member {member.id} in database...")
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
        if 'nfe' in self.end_criteria and self.end_criteria['nfe'] and self.nfe >= self.end_criteria['nfe']:
            return True
        if 'score' in self.end_criteria and self.end_criteria['score'] and any(member >= self.end_criteria['score'] for member in self.population.current):
            return True
        if all(self.is_member_finished(member) for member in self.population.current):
            return True
        return False

    def create_initial_generation(self) -> Generation:
        new_members = self.create_members(k=self.population_size)
        generation = Generation()
        trained_members = list(self.training_service.train(new_members, self.step_size))
        for member in trained_members:
            # log performance
            self._say(member.performance_details(), member)
            # Save member to database directory.
            self.update_database(member)
            generation.append(member)
        return generation

    def start(self, use_old = False):
        """
        Start global training procedure. Ends when end_criteria is met.
        """
        try:
            self.on_start()
            # start controller loop
            self._say("Starting training procedure...")
            if not use_old:
                self.train_synchronously()
            else:
                self.train_synchronously_old()
            # terminate worker processes
            self._say("finished.")
        except KeyboardInterrupt:
            self._say("interupted.")
        finally:
            self.on_end()

    def on_start(self):
        """Resets class properties, starts training service and cleans up temporary files."""
        # reset member state folder
        clear_member_states()
        # reset class properties
        self.nfe = 0
        self.generations = 0
        self.population = Population()
        # start training service
        self.training_service.start()

    def on_end(self):
        """Stops training service and cleans up temporary files."""
        # close training service
        self.training_service.stop()
        # reset member state folder
        clear_member_states()

    def train_synchronously(self):
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        but they are waiting for each other between each generation cycle.
        """
        self._say("Training initial generation...")
        self.population.append(self.create_initial_generation())
        while not self.is_population_finished():
            self._whisper("on generation start")
            self.evolver.on_generation_start(self.population.current, self._whisper)
            # create new generation
            new_generation = Generation()
            # generate new candidates
            new_candidates = self.evolver.on_evolve(copy.deepcopy(self.population.current), self._whisper)
            # 1. evolve, 2. train, 3. evaluate, 4. save
            for candidates in self.training_service.train(new_candidates, self.step_size):
                member = self.evolver.on_evaluation(candidates, self._whisper)
                self.nfe += 1 #if isinstance(candidates, Checkpoint) else len(candidates)
                # log performance
                self._say(member.performance_details(), member)
                # Save member to database directory.
                self.update_database(member)
                # write to tensorboard if enabled
                self._update_tensorboard(member)
                # Add member to generation.
                new_generation.append(member)
                self._whisper("awaiting next trained member...")
            self._whisper("on generation end")
            self.evolver.on_generation_end(new_generation, self._whisper)
            # add new generation
            self.population.append(new_generation)
            # perform garbage collection
            self._whisper("performing garbage collection...")
            self.remove_bad_member_states()
        self._say(f"end criteria has been reached.")

    def train_synchronously_old(self, eval_steps=1):
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        but they are waiting for each other between each generation cycle.
        """
        self._say("Training initial generation...")
        self.population.append(self.create_initial_generation())
        while not self.is_population_finished():
            self._whisper("on generation start")
            self.evolver.on_generation_start(self.population.current, self._whisper)
            # create new generation
            new_generation = Generation()
            # generate new candidates
            new_candidates = self.evolver.on_evolve(self.population.current, self._whisper)
            best_candidates = list()
            for candidates in self.training_service.train(new_candidates, eval_steps):
                member = self.evolver.on_evaluation(candidates, self._whisper)
                best_candidates.append(member)
                self.nfe += 1 #if isinstance(candidates, Checkpoint) else len(candidates)
            # train best
            for member in self.training_service.train(best_candidates, self.step_size - eval_steps):
                # log performance
                self._say(member.performance_details(), member)
                # Save member to database directory.
                self.update_database(member)
                # write to tensorboard if enabled
                self._update_tensorboard(member)
                # Add member to generation.
                new_generation.append(member)
                self._whisper("awaiting next trained member...")
            self._whisper("on generation end")
            self.evolver.on_generation_end(new_generation, self._whisper)
            # add new generation
            self.population.append(new_generation)
            # perform garbage collection
            self._whisper("performing garbage collection...")
            self.remove_bad_member_states()
        self._say(f"end criteria has been reached.")