import os
import sys
import math
import time
import random
import copy
import pickle
import shutil
import warnings
import itertools
from typing import List, Dict, Sequence, Iterator, Iterable
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
from pbt.worker_pool import WorkerPool
from pbt.member import Checkpoint, Population, Generation
from pbt.utils.date import get_datetime_string
from pbt.hyperparameters import DiscreteHyperparameter, Hyperparameters
from pbt.trainer import Trainer
from pbt.evaluator import Evaluator
from pbt.evolution import EvolveEngine
from pbt.database import Database
from pbt.garbage import GarbageCollector
from pbt.step import Step

class Controller(object):
    def __init__(
            self, population_size: int, hyper_parameters: Hyperparameters, trainer: Trainer, evaluator: Evaluator, evolver: EvolveEngine,
            loss_metric: str, eval_metric: str, loss_functions: dict, database : Database, tester: Evaluator = None,
            step_size: int = 1, eval_steps: int = 0, end_criteria: dict = {'score': 100.0}, detect_NaN: bool = False, history_limit: int = None,
            devices: List[str] = ['cpu'], n_jobs: int = -1, verbose: int = 1, logging: bool = True, tensorboard: SummaryWriter = None):
        if not isinstance(step_size, int) or step_size < 1: 
            raise Exception(f"step size must be of type {int} and 1 or higher.")
        if not isinstance(eval_steps, int) or eval_steps >= step_size:
            raise Exception(f"eval steps must be of type {int} and lower than step size.")
        self.population_size = population_size
        self.database = database
        self.trainer = trainer
        self.evaluator = evaluator
        self.tester = tester
        self.evolver = evolver
        self.hyper_parameters = hyper_parameters
        self.worker_pool = WorkerPool(devices=devices, n_jobs=n_jobs, verbose=max(verbose - 2, 0))
        self.garbage_collector = GarbageCollector(database=database, history_limit=history_limit if history_limit and history_limit > 2 else 2, verbose=verbose-2)
        self.step_size = step_size
        self.eval_steps = eval_steps
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss_functions = loss_functions
        self.end_criteria = end_criteria
        self.detect_NaN = detect_NaN
        self.verbose = verbose
        self.logging = logging
        self.nfe = 0
        self.n_generations = 0
        self.__tensorboard = tensorboard

    def __create_message(self, message : str, tag : str = None) -> str:
        time = get_datetime_string()
        generation = f"G{len(self.n_generations):03d}"
        nfe = f"({self.nfe}/{self.end_criteria['nfe']})" if 'nfe' in self.end_criteria and self.end_criteria['nfe'] else self.nfe
        return f"{time} {nfe} {generation}{f' {tag}' if tag else ''}: {message}"

    def __say(self, message : str, member : Checkpoint = None) -> None:
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 0."""
        self.__register(message=message, verbosity=0, member=member)

    def __whisper(self, message : str, member = None) -> None:
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 1."""
        self.__register(message=message, verbosity=1, member=member)

    def __register(self, message : str, verbosity : int, member : Checkpoint = None) -> None:
        """Logs and prints the provided message in the appropriate syntax. If a member is provided, the message is attached to that member."""
        print_tag = member if member else self.__class__.__name__
        file_tag = f"member_{member.id}" if member else self.__class__.__name__
        full_message = self.__create_message(message, tag=print_tag)
        self.__print(message=full_message, verbosity=verbosity)
        self.__log_to_file(message=full_message, tag=file_tag)
        self.__log_to_tensorboard(message=full_message, tag=file_tag, global_steps=member.steps if member else None)
    
    def __print(self, message : str, verbosity : int) -> None:
        if self.verbose > verbosity:
            print(message)

    def __log_to_file(self, message : str, tag : str) -> None:
        if not self.logging:
            return
        with self.database.create_file(tag='logs', file_name=f"{tag}_log.txt").open('a+') as file:
            file.write(message + '\n')
    
    def __log_to_tensorboard(self, message : str, tag : str, global_steps : int = None) -> None:
        if not self.__tensorboard:
            return
        self.__tensorboard.add_text(tag=tag, text_string=message, global_step=global_steps)

    def __update_tensorboard(self, member : Checkpoint) -> None:
        """Plots member data to tensorboard"""
        if not self.__tensorboard:
            return       
        # plot eval metrics
        for eval_metric_group, eval_metrics in member.loss.items():
            for metric_name, metric_value in eval_metrics.items():
                self.__tensorboard.add_scalar(
                    tag=f"metrics/{eval_metric_group}_{metric_name}/{member.id:03d}",
                    scalar_value=metric_value,
                    global_step=member.steps)
        # plot time
        for time_type, time_value in member.time.items():
            self.__tensorboard.add_scalar(
                tag=f"time/{time_type}/{member.id:03d}",
                scalar_value=time_value,
                global_step=member.steps)
        # plot hyper-parameters
        for hparam_name, hparam in member.parameters.items():
            self.__tensorboard.add_scalar(
                tag=f"hyperparameters/{hparam_name}/{member.id:03d}",
                scalar_value=hparam.normalized if isinstance(hparam, DiscreteHyperparameter) else hparam.value,
                global_step=member.steps)

    def __create_member(self, id) -> Checkpoint:
        """Create a member object"""
        # create new member object
        member = Checkpoint(
            id=id,
            parameters=copy.deepcopy(self.hyper_parameters),
            loss_metric=self.loss_metric,
            eval_metric=self.eval_metric,
            minimize=self.loss_functions[self.eval_metric].minimize)
        # process new member with evolver
        self.evolver.on_spawn(member, self.__whisper)
        return member

    def __create_members(self, k : int) -> List[Checkpoint]:
        members = list()
        for id in range(k):
            members.append(self.__create_member(id))
        return members

    def __update_database(self, member : Checkpoint) -> None:
        """Updates the database stored in files."""
        self.__whisper(f"updating member {member.id} in database...")
        self.database.update(member.id, member.steps, member)

    def __is_member_finished(self, member : Checkpoint) -> bool:
        """With the end_criteria, check if the provided member is finished training."""
        if 'steps' in self.end_criteria and self.end_criteria['steps'] and member.steps >= self.end_criteria['steps']:
            # the number of steps is equal or above the given treshold
            return True
        return False
    
    def __is_population_finished(self, generation: Generation) -> bool:
        """
        With the end_criteria, check if the entire population is finished
        by inspecting the provided member.
        """
        if 'nfe' in self.end_criteria and self.end_criteria['nfe'] and self.nfe >= self.end_criteria['nfe']:
            return True
        if 'score' in self.end_criteria and self.end_criteria['score'] and any(member >= self.end_criteria['score'] for member in generation):
            return True
        if all(self.__is_member_finished(member) for member in generation):
            return True
        return False

    def __create_initial_generation(self) -> Generation:
        new_members = self.__create_members(k=self.population_size)
        return Generation(new_members)

    def start(self) -> List[Checkpoint]:
        """
        Start global training procedure. Ends when end_criteria is met.
        """
        try:
            self.__on_start()
            # start controller loop
            self.__say("Starting training procedure...")
            return self.__train_synchronously()
        except KeyboardInterrupt:
            self.__say("interupted.")
        finally:
            self.__say("finished.")
            self.__on_stop()

    def __on_start(self) -> None:
        """Resets class properties, starts training service and cleans up temporary files."""
        # reset class properties
        self.nfe = 0
        self.n_generations = 0
        # start training service
        self.worker_pool.start()

    def __on_stop(self) -> None:
        """Stops training service and cleans up temporary files."""
        # close training service
        self.worker_pool.stop()
    
    def __train(self, checkpoints : Iterable[Checkpoint], train_steps: int, train_shuffle: bool = False):
        step = Step(trainer=self.trainer, evaluator=self.evaluator, train_step_size=train_steps, eval_step_size=None, train_shuffle=train_shuffle, eval_shuffle=False)
        yield from self.worker_pool.imap(step, checkpoints)

    def __eval(self, checkpoints : Iterable[Checkpoint], eval_steps: int, train_shuffle: bool = False):
        step = Step(trainer=self.trainer, evaluator=self.evaluator, tester=self.tester, train_step_size = eval_steps, eval_step_size=eval_steps, train_shuffle=train_shuffle, eval_shuffle=True)
        yield from self.worker_pool.imap(step, checkpoints)

    def __train_synchronously(self) -> List[Checkpoint]:
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        but they are waiting for each other between each generation cycle.
        """
        self.__say("Training initial generation...")
        previous_generation = None
        generation = self.__create_initial_generation()
        while not self.__is_population_finished(generation):
            self.__whisper("on generation start")
            self.evolver.on_generation_start(generation, self.__whisper)
            # create new generation
            new_generation = Generation()
            # train candidates for n steps
            for candidates in self.__train(generation, self.step_size - self.eval_steps):
                member = self.evolver.on_evaluate(candidates, self.__whisper) if self.eval_steps <= 0 else candidates
                self.nfe += 1 #if isinstance(candidates, Checkpoint) else len(candidates)
                # log performance
                self.__say(member.performance_details(), member)
                # Save member to database directory.
                self.__update_database(member)
                # write to tensorboard if enabled
                self.__update_tensorboard(member)
                # Add member to generation.
                new_generation.append(member)
                self.__whisper("awaiting next trained member...")
            # generate new candidates
            new_candidates = self.evolver.on_evolve(new_generation, self.__whisper)
            # test candidates with a smaller eval step
            if self.eval_steps > 0:
                best_candidates = list()
                for candidates in self.__eval(new_candidates, self.eval_steps):
                    member = self.evolver.on_evaluate(candidates, self.__whisper)
                    best_candidates.append(member)
                new_candidates = best_candidates
            self.__whisper("on generation end")
            self.evolver.on_generation_end(new_generation, self.__whisper)
            # add new generation
            previous_generation = generation
            generation = new_generation
            # perform garbage collection
            self.__whisper("performing garbage collection...")
            self.garbage_collector.collect(previous_generation)
        self.__say(f"end criteria has been reached.")
        return list(generation)

        # step
        # eval
        # if ready do
        # mutate
        # eval
        # update population