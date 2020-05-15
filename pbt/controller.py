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
from abc import abstractmethod
from typing import List, Dict, Sequence, Iterator, Iterable, Tuple
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
from pbt.member import Checkpoint, Generation
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
            end_criteria: dict = {'score': 100.0}, history_limit: int = None,
            devices: List[str] = ['cpu'], n_jobs: int = -1, verbose: int = 1, logging: bool = True, tensorboard: SummaryWriter = None):
        self.population_size = population_size
        self.database = database
        self.trainer = trainer
        self.evaluator = evaluator
        self.tester = tester
        self.evolver = evolver
        self.hyper_parameters = hyper_parameters
        self.worker_pool = WorkerPool(devices=devices, n_jobs=n_jobs, verbose=max(verbose - 2, 0))
        self.garbage_collector = GarbageCollector(database=database, history_limit=history_limit if history_limit and history_limit > 2 else 2, verbose=verbose-2)
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss_functions = loss_functions
        self.end_criteria = end_criteria
        self.verbose = verbose
        self.logging = logging
        self.nfe = 0
        self.n_generations = 0
        self.__tensorboard = tensorboard

    def __create_message(self, message : str, tag : str = None) -> str:
        time = get_datetime_string()
        generation = f"G{self.n_generations:03d}"
        nfe = f"({self.nfe}/{self.end_criteria['nfe']})" if 'nfe' in self.end_criteria and self.end_criteria['nfe'] else self.nfe
        return f"{time} {nfe} {generation}{f' {tag}' if tag else ''}: {message}"

    def _say(self, message : str, member : Checkpoint = None) -> None:
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 0."""
        self.__register(message=message, verbosity=0, member=member)

    def _whisper(self, message : str, member = None) -> None:
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
        """Plots member data to tensorboard."""
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
        """Create a member object."""
        # create new member object
        member = Checkpoint(
            id=id,
            parameters=copy.deepcopy(self.hyper_parameters),
            loss_metric=self.loss_metric,
            eval_metric=self.eval_metric,
            minimize=self.loss_functions[self.eval_metric].minimize)
        # process new member with evolver
        self.evolver.on_spawn(member, self._whisper)
        return member

    def __create_members(self, k : int) -> List[Checkpoint]:
        members = list()
        for id in range(k):
            members.append(self.__create_member(id))
        return members

    def __update_database(self, member : Checkpoint) -> None:
        """Updates the database stored in files."""
        self._whisper(f"updating member {member.id} in database...")
        self.database.update(member.id, member.steps, member)

    def __is_member_finished(self, member : Checkpoint) -> bool:
        """With the end_criteria, check if the provided member is finished training."""
        if 'steps' in self.end_criteria and self.end_criteria['steps'] and member.steps >= self.end_criteria['steps']:
            # the number of steps is equal or above the given treshold
            return True
        return False
    
    def __is_finished(self, generation: Generation) -> bool:
        """With the end_criteria, check if the generation is finished by inspecting the provided member."""
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

    def start(self) -> Generation:
        """Start global training procedure. Ends when end_criteria is met."""
        try:
            self.__on_start()
            # start controller loop
            self._say("Starting training procedure...")
            return self.__train_synchronously()
        except KeyboardInterrupt:
            self._say("interupted.")
        finally:
            self._say("finished.")
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
    
    @abstractmethod
    def _create_next_generation(self, generation):
        raise NotImplementedError()

    def __train_synchronously(self) -> Generation:
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        but they are waiting for each other between each generation cycle.
        """
        self._say("Creating initial generation...")
        generation = self.__create_initial_generation()
        while not self.__is_finished(generation):
            self._whisper("on generation start...")
            self.evolver.on_generation_start(generation, self._whisper)
            generation = self._create_next_generation(generation)
            self._whisper("on generation end...")
            self.evolver.on_generation_end(generation, self._whisper)
            # Save member to database directory.
            [self.__update_database(member) for member in generation]
            # write to tensorboard if enabled
            [self.__update_tensorboard(member) for member in generation]
            # perform garbage collection
            self._whisper("performing garbage collection...")
            self.garbage_collector.collect()
            # increment number of generations
            self.n_generations += 1
        self._say(f"end criteria has been reached.")
        return generation

class PBTController(Controller):
    def __init__(self, step_size: int, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(step_size, int) or step_size < 1: 
            raise Exception(f"step size must be of type {int} and 1 or higher.")
        self.step_size = step_size

    def _create_next_generation(self, generation):
        self._whisper("training generation...")
        trained_members = Generation()
        # 1. STEP: train candidates for n train steps
        for trained_member in self.__train(generation):
            # log performance
            self._say(trained_member.performance_details(), trained_member)
            # Add member to generation.
            trained_members.append(trained_member)
        # 2. EVOLVE: generate candidates
        self._whisper("evolving generation...")
        new_generation = Generation()
        for trial_member in self.evolver.on_evolve(trained_members, self._whisper):
            best_member = self.evolver.on_evaluate(trial_member, self._whisper)
            new_generation.append(best_member)
            self.nfe += 1
        return new_generation

    def __train(self, checkpoints : Iterable[Tuple[Checkpoint]]):
        step = Step(trainer=self.trainer, evaluator=self.evaluator, tester=self.tester, train_step_size=self.step_size, eval_step_size=None, train_shuffle=False, eval_shuffle=False)
        self._whisper(f"queing generation for training with {self.step_size} steps...")
        yield from self.worker_pool.imap(step, checkpoints)

class DEController(Controller):
    def __init__(self, step_size: int, eval_steps: int = 0, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(step_size, int) or step_size < 0: 
            raise Exception(f"step size must be of type {int} and 1 or higher.")
        if not isinstance(eval_steps, int) or not(0 < eval_steps <= step_size):
            raise Exception(f"eval steps must be of type {int} and equal or lower than zero.")
        self.step_size = step_size
        self.eval_steps = eval_steps
        self.__has_pretrained = False

    def _create_next_generation(self, generation: Generation):
        # 0. PRE-TRAIN
        if not self.__has_pretrained:
            self._whisper("pre-training generation...")
            generation = self._step(generation, self.step_size)
            self.__has_pretrained = True
        # 1. ON EVOLVE
        self._whisper("evolving generation...")
        new_candidates = self.evolver.on_evolve(generation, self._whisper)
        new_generation = Generation()
        # 2. ON EVAL
        self._whisper("evaluating generation...")
        for candidates in self.__eval(new_candidates):
            best_member = self.evolver.on_evaluate(candidates, self._whisper)
            new_generation.append(best_member)
            self.nfe += 1
        # 3. ON TRAIN
        self._whisper("training generation...")
        trained_members = self._step(new_generation, self.step_size - self.eval_steps)
        return trained_members

    def __train(self, checkpoints: Iterable[Tuple[Checkpoint]], n_steps: int):
        if n_steps > 0:
            step = Step(trainer=self.trainer, evaluator=self.evaluator, tester=self.tester, train_step_size=n_steps, eval_step_size=None, train_shuffle=False, eval_shuffle=False)
            self._whisper(f"queing generation for training with {n_steps} steps...")
            yield from self.worker_pool.imap(step, checkpoints)
        else:
            yield from checkpoints

    def __eval(self, checkpoints: Iterable[Tuple[Checkpoint]]):
        if self.eval_steps > 0:
            step = Step(trainer=self.trainer, evaluator=self.evaluator, train_step_size = self.eval_steps, eval_step_size=self.eval_steps, train_shuffle=False, eval_shuffle=True)
            self._whisper(f"queing generation for evaluation with {self.eval_steps} steps...")
            yield from self.worker_pool.imap(step, checkpoints)
        else:
            yield from checkpoints
    
    def _step(self, generation: Generation, n_steps: int):
        self._whisper("training generation...")
        trained_members = Generation()
        for trained_member in self.__train(generation, n_steps):
            self._say(trained_member.performance_details(), trained_member)
            trained_members.append(trained_member)
        return trained_members

