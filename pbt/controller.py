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
from datetime import datetime, timedelta
from abc import abstractmethod
from typing import List, Dict, Sequence, Iterator, Iterable, Tuple, Callable
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
            self, manager, population_size: int, hyper_parameters: Hyperparameters, evolver: EvolveEngine,
            loss_metric: str, eval_metric: str, loss_functions: dict, database : Database, end_criteria: dict = {'score': 100.0}, history_limit: int = None,
            devices: List[str] = ['cpu'], n_jobs: int = -1, verbose: int = 1, logging: bool = True, tensorboard: SummaryWriter = None):
        self.population_size = population_size
        self.database = database
        self.evolver = evolver
        self.hyper_parameters = hyper_parameters
        self.worker_pool = WorkerPool(manager=manager, devices=devices, n_jobs=n_jobs, verbose=max(verbose - 2, 0))
        self.garbage_collector = GarbageCollector(database=database, history_limit=history_limit if history_limit and history_limit > 2 else 2, verbose=verbose-2)
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss_functions = loss_functions
        self.end_criteria = end_criteria
        self.verbose = verbose
        self.logging = logging
        self.__n_steps = 0
        self.__n_generations = 0
        self.__start_time = None
        self.__tensorboard = tensorboard

    @property
    def end_time(self):
        if self.__start_time is None or 'time' not in self.end_criteria or not self.end_criteria['time']:
            return None
        return self.__start_time + timedelta(minutes=self.end_criteria['time'])

    def __create_message(self, message : str, tag : str = None) -> str:
        time = get_datetime_string()
        generation = f"G{self.__n_generations:03d}"
        n_steps = f"({self.__n_steps}/{self.end_criteria['steps']})" if 'steps' in self.end_criteria and self.end_criteria['steps'] else self.__n_steps
        return f"{time} {n_steps} {generation}{f' {tag}' if tag else ''}: {message}"

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
        return member

    def __create_members(self, k : int) -> List[Checkpoint]:
        members = list()
        for id in range(k):
            members.append(self.__create_member(id))
        return members

    def __create_initial_generation(self) -> Generation:
        new_members = self.__create_members(k=self.population_size)
        # process new member with evolver
        generation = self.evolver.spawn(new_members)
        return generation

    def __update_database(self, member : Checkpoint) -> None:
        """Updates the database stored in files."""
        self._whisper(f"updating member {member.id} in database...")
        self.database.update(member.id, member.steps, member)
    
    def __is_score_end(self, generation):
        return 'score' in self.end_criteria and self.end_criteria['score'] and any(member >= self.end_criteria['score'] for member in generation)

    def __is_steps_end(self):
        return 'steps' in self.end_criteria and self.end_criteria['steps'] and self.__n_steps >= self.end_criteria['steps']

    def __is_time_end(self):
        return 'time' in self.end_criteria and self.end_criteria['time'] and datetime.now() >= self.end_time

    def __is_finished(self, generation: Generation) -> bool:
        """With the end_criteria, check if the generation is finished by inspecting the provided member."""
        is_met = False
        if self.__is_score_end(generation):
            self._say("Score criterium has been met!")
            is_met = True
        if self.__is_steps_end():
            self._say("Step criterium has been met!")
            is_met = True
        if self.__is_time_end():
            self._say("Time criterium has been met!")
            is_met = True
        return is_met

    def start(self) -> Generation:
        """Start global training procedure. Ends when end_criteria is met."""
        try:
            self.__on_start()
            # start controller loop
            self._say("Starting training procedure...")
            return self.__train_synchronously()
        except KeyboardInterrupt:
            self._say("interupted.")
        except Exception:
            self._say("Exception not handled!")
            raise
        finally:
            self._say("finished.")
            self.__on_stop()

    def __on_start(self) -> None:
        """Resets class properties, starts training service and cleans up temporary files."""
        # reset class properties
        self.__start_time = datetime.now()
        self.__n_steps = 0
        self.__n_generations = 0
        # start training service
        self.worker_pool.start()

    def __on_stop(self) -> None:
        """Stops training service and cleans up temporary files."""
        # close training service
        self.worker_pool.stop()

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
            self.evolver.on_generation_start(generation)
            for member in self.worker_pool.imap(self.create_procedure(generation), generation):
                self.__n_steps += 1
                self._say(member.performance_details(), member)
                generation.update(member)
                # Save member to database directory.
                self.__update_database(member)
                # write to tensorboard if enabled
                self.__update_tensorboard(member)
                continue
            self._whisper("on generation end...")
            self.evolver.on_generation_end(generation)
            # perform garbage collection
            self._whisper("performing garbage collection...")
            self.garbage_collector.collect()
            # increment number of generations
            self.__n_generations += 1
        self._say(f"end criteria has been reached.")
        return generation

    @abstractmethod
    def create_procedure(self, generation) -> Callable[[Generation], Checkpoint]:
        raise NotImplementedError()

class PBTProcedure:
    def __init__(self, generation: Generation, evolver: EvolveEngine, train_function):
        self.generation = generation
        self.evolver = evolver
        self.train_function = train_function

    def __call__(self, member: Checkpoint, device: str):
        new_member = self.evolver.mutate(
            member=member,
            generation=self.generation,
            train_function=partial(self.train_function, device=device))
        return new_member
        
class PBTController(Controller):
    def __init__(self, trainer: Trainer, evaluator: Evaluator, step_size: int, tester: Evaluator = None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(step_size, int) or step_size < 0: 
            raise Exception(f"step size must be of type {int} and 1 or higher.")
        self.train_step = Step(
            train_function=partial(trainer, step_size=step_size),
            eval_function=evaluator,
            verbose=self.verbose > 3)
        if tester is not None:
            self.train_step.functions['test_function'] = tester

    def create_procedure(self, generation: Generation):
        return PBTProcedure(generation=generation, evolver=self.evolver, train_function=self.train_step)

class DEProcedure:
    def __init__(self, generation: Generation, evolver: EvolveEngine, train_function, fitness_function, test_function=None):
        self.generation = generation
        self.train_function = train_function
        self.fitness_function = fitness_function
        self.test_function = test_function
        self.evolver = evolver

    def __call__(self, member: Checkpoint, device: str):
        trained_member = self.train_function(member, device=device)
        new_member = self.evolver.mutate(
            parent=trained_member,
            generation=self.generation,
            fitness_function=partial(self.fitness_function, device=device))
        # add loss
        for group in new_member.loss:
            new_member.loss[group] = {k: (v + new_member.loss[group][k])/2.0 for k, v in trained_member.loss[group].items()}
        # measure test set performance if available
        if self.test_function is not None:
            self.test_function(new_member, device)
        return new_member
        
class DEController(Controller):
    def __init__(self, trainer: Trainer, evaluator: Evaluator, step_size: int, eval_steps: int, tester: Evaluator = None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(step_size, int) or step_size < 0: 
            raise Exception(f"step size must be of type {int} and 1 or higher.")
        if not isinstance(eval_steps, int) or not(0 < eval_steps <= step_size):
            raise Exception(f"eval steps must be of type {int} and equal or lower than zero.")
        self.train_step = Step(
            train_function=partial(trainer, step_size=step_size - eval_steps),
            eval_function=evaluator,
            verbose=self.verbose > 3)
        self.fitness_step = Step(
            train_function=partial(trainer, step_size=eval_steps),
            eval_function=partial(evaluator, step_size=eval_steps, shuffle=True),
            verbose=self.verbose > 3)
        self.test_step = Step(
            test_function=tester,
            verbose=self.verbose > 3) if tester else None

    def create_procedure(self, generation: Generation):
        return DEProcedure(generation=generation, evolver=self.evolver, train_function=self.train_step, fitness_function=self.fitness_step, test_function=self.test_step)


