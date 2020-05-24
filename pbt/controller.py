import os
import sys
import math
import time
import dill
import random
import copy
import pickle
import shutil
import warnings
import itertools
from datetime import datetime, timedelta
from abc import abstractmethod
from typing import List, Dict, Sequence, Iterator, Iterable, Tuple, Callable, Generator
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
from pbt.worker import DeviceCallable
from pbt.worker_pool import WorkerPool
from pbt.member import Checkpoint, Generation
from pbt.utils.date import get_datetime_string
from pbt.hyperparameters import DiscreteHyperparameter, Hyperparameters
from pbt.nn import Trainer, Evaluator, RandomFitnessApproximation
from pbt.evolution import ExploitAndExplore, DifferentialEvolveEngine
from pbt.database import Database
from pbt.garbage import GarbageCollector

class Controller(object):
    def __init__(
            self, manager, population_size: int, hyper_parameters: Hyperparameters, loss_metric: str, eval_metric: str, loss_functions: dict, database : Database, end_criteria: dict = {'score': 100.0}, history_limit: int = None,
            devices: List[str] = ['cpu'], n_jobs: int = -1, verbose: int = 1, logging: bool = True, tensorboard: SummaryWriter = None):
        self.population_size = population_size
        self.database = database
        self.hyper_parameters = hyper_parameters
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss_functions = loss_functions
        self.end_criteria = end_criteria
        self.verbose = verbose
        self.logging = logging
        self._manager = manager
        self._worker_pool = WorkerPool(manager=manager, devices=devices, n_jobs=n_jobs, verbose=max(verbose - 2, 0))
        self.__garbage_collector = GarbageCollector(database=database, history_limit=history_limit if history_limit and history_limit > 2 else 2, verbose=verbose-2)
        self.__n_generations = 0
        self.__start_time = None
        self.__tensorboard = tensorboard

    @property
    def end_time(self):
        if self.__start_time is None or 'time' not in self.end_criteria or not self.end_criteria['time']:
            return None
        return self.__start_time + timedelta(minutes=self.end_criteria['time'])

    @abstractmethod
    def _print_prefix(self) -> str:
        raise NotImplementedError

    def __create_message(self, message : str, tag : str = None) -> str:
        time = get_datetime_string()
        generation = f"G{self.__n_generations:03d}"
        prefixes = ' '.join(prefix for prefix in (time, generation, self._print_prefix(), tag) if prefix is not None)
        return f"{prefixes}: {message}"

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
        return new_members

    def __update_database(self, member : Checkpoint) -> None:
        """Updates the database stored in files."""
        self.database.update(member.id, member.steps, member)
    
    def __is_score_end(self, generation):
        return 'score' in self.end_criteria and self.end_criteria['score'] and any(member >= self.end_criteria['score'] for member in generation)

    def __is_time_end(self):
        return 'time' in self.end_criteria and self.end_criteria['time'] and datetime.now() >= self.end_time

    def __is_n_generations_end(self):
        return 'generations' in self.end_criteria and self.end_criteria['generations'] and self.__n_generations >= self.end_criteria['generations']

    def _is_finished(self, generation: Generation) -> bool:
        """With the end_criteria, check if the generation is finished by inspecting the provided member."""
        if self.__is_score_end(generation):
            self._say("Score criterium has been met!")
            return True
        if self.__is_time_end():
            self._say("Time criterium has been met!")
            return True
        if self.__is_n_generations_end():
            self._say("Number of generations criterium has been met!")
            return True
        return False

    def start(self) -> Checkpoint:
        """Start global training procedure. Ends when end_criteria is met."""
        try:
            self._on_start()
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
            self._on_stop()

    def _on_start(self) -> None:
        """Resets class properties, starts training service and cleans up temporary files."""
        # reset class properties
        self.__start_time = datetime.now()
        self.__n_generations = 0
        # start training service
        self._worker_pool.start()

    def _on_stop(self) -> None:
        """Stops training service and cleans up temporary files."""
        # close training service
        self._worker_pool.stop()

    def __train_synchronously(self) -> Generation:
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        but they are waiting for each other between each generation cycle.
        """
        self._say("Creating initial generation...")
        initial_members = self.__create_initial_generation()
        for generation in self._train(initial_members):
            # Save member to database directory.
            self._whisper(f"saving members to database...")
            [self.__update_database(member) for member in generation]
            # write to tensorboard if enabled
            [self.__update_tensorboard(member) for member in generation]
            # perform garbage collection
            self._whisper("performing garbage collection...")
            self.__garbage_collector.collect()
            # increment number of generations
            self.__n_generations += 1
        self._say(f"end criteria has been reached.")
        return max(generation)

    @abstractmethod
    def _train(self, initial: Iterable[Checkpoint]) -> Generator[List[Checkpoint], None, None]:
        raise NotImplementedError()

class PBTController(Controller):
    def __init__(self, evolver: ExploitAndExplore, model_class, optimizer_class, datasets, batch_size: int, train_steps: int, **kwargs):
        super().__init__(**kwargs)
        # evolver
        self.evolver = evolver
        self.evolver.verbose = self.verbose > 2
        # create training function
        self.train_function = Trainer(
            model_class = model_class, optimizer_class = optimizer_class, train_data = datasets.train,
            loss_functions = self.loss_functions, loss_metric = self.loss_metric,
            batch_size = batch_size, step_size=train_steps, verbose=self.verbose > 5)
        # create evaluation function
        self.eval_function = Evaluator(
            model_class = model_class, test_data = datasets.eval, loss_functions=self.loss_functions,
            batch_size = batch_size, loss_group = 'eval', verbose=self.verbose > 5)
        # creating test function if test set is provided
        self.test_function = Evaluator(
            model_class = model_class, test_data = datasets.test, loss_functions=self.loss_functions,
            batch_size = batch_size, loss_group = 'test', verbose=self.verbose > 5) if datasets.test else None
        self.__n_steps = 0

    def _print_prefix(self) -> str:
        if self.end_criteria['steps']:
            return f"({self.__n_steps}/{self.end_criteria['steps']})"
        else:
            return ""

    def _on_start(self) -> None:
        self.__n_steps = 0
        super()._on_start()

    def __is_steps_end(self):
        return 'steps' in self.end_criteria and self.end_criteria['steps'] and self.__n_steps >= self.end_criteria['steps']

    def _is_finished(self, generation: Generation):
        if super()._is_finished(generation):
            return True
        if self.__is_steps_end():
            self._say("Step criterium has been met!")
            return True
        return False

    def _train(self, initial: Iterable[Checkpoint]) -> Generator[List[Checkpoint], None, None]:
        # spawn members
        spawned_members = self.evolver.spawn(initial)
        # create generation
        generation = Generation(dict_constructor=self.manager.dict, members=spawned_members)
        # create and start procedure
        procedure = self.create_procedure(generation)
        [self._worker_pool.apply_async(procedure, member) for member in generation]
        # loop until finished
        while not self._is_finished(generation):
            # get member and update generation
            member = self._worker_pool.get()
            # increment steps
            self.__n_steps += 1
            # report member performance
            self._say(member.performance_details(), member)
            # queue member for next training
            self._worker_pool.apply_async()
            # yield generation if N members has been processed
            if self.__n_steps % len(generation) == 0:
                yield list(generation)
        yield list(generation)

    def create_procedure(self, generation: Generation):
        return PBTProcedure(generation=generation, evolver=self.evolver, train_function=self.train_function, eval_function=self.eval_function,
            test_function=self.test_function, verbose=self.verbose > 3)

class PBTProcedure(DeviceCallable):
    def __init__(self, generation: Generation, evolver: ExploitAndExplore, train_function, eval_function, test_function=None, verbose: bool = False):
        super().__init__(verbose)
        self.generation = generation
        self.evolver = evolver
        self.train_function = train_function
        self.eval_function = eval_function
        self.test_function = test_function

    def __call__(self, member: Checkpoint, device: str) -> Checkpoint:
        self._print(f"training member {member.id}...")
        self.train_function(member, device)
        self._print(f"evaluating member {member.id}...")
        self.eval_function(member, device)
        self.generation.update(member)
        # exploit and explore
        self._print(f"mutating member {member.id}...")
        member = self.evolver.mutate(member=member, generation=self.generation)
        # measure test set performance if available
        if self.test_function is not None:
            self._print(f"testing member {member.id}...")
            self.test_function(member, device)
        # update generation
        self.generation.update(member)
        return member

class DEController(Controller):
    def __init__(self, evolver: DifferentialEvolveEngine, model_class, optimizer_class, datasets, batch_size: int, train_steps: int, fitness_steps: int, **kwargs):
        super().__init__(**kwargs)
        # evolver
        self.evolver = evolver
        self.evolver.verbose = self.verbose > 2
        # create training function
        self.train_function = Trainer(
            model_class = model_class, optimizer_class = optimizer_class, train_data = datasets.train,
            loss_functions = self.loss_functions, loss_metric = self.loss_metric,
            batch_size = batch_size, step_size=train_steps, verbose=self.verbose > 5)
        # create evaluation function
        self.eval_function = Evaluator(
            model_class = model_class, test_data = datasets.eval, loss_functions=self.loss_functions,
            batch_size = batch_size, loss_group = 'eval', verbose=self.verbose > 5)
        # creating fitness function
        self.partial_fitness_function = partial(RandomFitnessApproximation, model_class=model_class, optimizer_class=optimizer_class, train_data=datasets.train, test_data=datasets.eval,
            loss_functions=self.loss_functions, loss_metric=self.loss_metric, batch_size=batch_size, batches=fitness_steps, verbose=self.verbose > 5)
        # creating test function if test set is provided
        self.test_function = Evaluator(
            model_class = model_class, test_data = datasets.test, loss_functions=self.loss_functions,
            batch_size = batch_size, loss_group = 'test', verbose=self.verbose > 5) if datasets.test else None
        self.__n_steps = 0

    def _print_prefix(self) -> str:
        if self.end_criteria['steps']:
            return f"({self.__n_steps}/{self.end_criteria['steps']})"
        else:
            return ""

    def _on_start(self) -> None:
        self.__n_steps = 0
        super()._on_start()

    def __is_steps_end(self):
        return 'steps' in self.end_criteria and self.end_criteria['steps'] and self.__n_steps >= self.end_criteria['steps']

    def _is_finished(self, generation: Generation):
        if super()._is_finished(generation):
            return True
        if self.__is_steps_end():
            self._say("Step criterium has been met!")
            return True
        return False

    def _train(self, initial: Iterable[Checkpoint]) -> Generator[List[Checkpoint], None, None]:
        # spawn members
        spawned_members = self.evolver.spawn(initial)
        # create generation
        generation = Generation(dict_constructor=self.manager.dict, members=spawned_members)
        while not self._is_finished(generation):
            # increment n steps
            self._whisper("on generation start...")
            self.evolver.on_generation_start(generation)
            for member in self.__mutate_asynchronously(generation):
                # report member performance
                self._say(member.performance_details(), member)
                self.__n_steps += 1
            self._whisper("on generation end...")
            self.evolver.on_generation_end(generation)
            yield list(generation)

    def __mutate_asynchronously(self, generation: Generation):
        procedure = DEProcedure(generation=generation, evolver=self.evolver, train_function=self.train_function, eval_function=self.eval_function,
            fitness_function=self.partial_fitness_function(), test_function=self.test_function, verbose=self.verbose > 3)
        yield from self._worker_pool.imap(procedure, generation)

class DEProcedure(DeviceCallable):
    def __init__(self, generation: Generation, evolver: DifferentialEvolveEngine, train_function, eval_function, fitness_function, test_function=None, verbose: bool = False):
        super().__init__(verbose)
        self.generation = generation
        self.evolver = evolver
        self.train_function = train_function
        self.eval_function = eval_function
        self.fitness_function = fitness_function
        self.test_function = test_function

    def __call__(self, member: Checkpoint, device: str) -> Checkpoint:
        self._print(f"training member {member.id}...")
        self.train_function(member, device)
        self._print(f"evaluating member {member.id}...")
        self.eval_function(member, device)
        self._print(f"mutating member {member.id}...")
        new_member = self.evolver.mutate(
            parent=member,
            generation=self.generation,
            fitness_function=partial(self.fitness_function, device=device))
        # measure test set performance if available
        if self.test_function is not None:
            self._print(f"testing member {member.id}...")
            self.test_function(member, device)
        # update generation
        self.generation.update(member)
        return member