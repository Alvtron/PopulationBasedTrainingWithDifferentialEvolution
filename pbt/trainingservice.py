import os
import sys
import math
import itertools
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Sequence, Callable
from functools import partial
from multiprocessing.context import BaseContext
from multiprocessing.pool import ThreadPool

import pbt.member
from .member import Checkpoint, Population, Generation, clear_member_states
from .fitness import train_and_evaluate
from .utils.date import get_datetime_string
from .hyperparameters import DiscreteHyperparameter, Hyperparameters
from .trainer import Trainer
from .evaluator import Evaluator
from .evolution import EvolveEngine
from .database import Database

class Job:
    def __init__(self, checkpoints : Tuple[Checkpoint], step_size : int, device : str):
        if isinstance(checkpoints, Sequence) and all(isinstance(checkpoint, Checkpoint) for checkpoint in checkpoints):
            self.checkpoints = checkpoints
        elif isinstance(checkpoints, Checkpoint):
            self.checkpoints = tuple([checkpoints])
        else:
            raise TypeError
        if not isinstance(step_size, int):
            raise TypeError
        if not isinstance(device, str):
            raise TypeError
        self.step_size = step_size
        self.device = device

class FitnessFunction(object):
    def __init__(self, trainer : Trainer, evaluator : Evaluator):
        self.trainer = trainer
        self.evaluator = evaluator

    def __call__(self, job : Job) -> Tuple[Checkpoint, ...]:
        fit_function = partial(train_and_evaluate, trainer=self.trainer, evaluator=self.evaluator, step_size=job.step_size, device=job.device)
        if not job.checkpoints:
            raise ValueError("No checkpoints available in job-object.")
        elif len(job.checkpoints) == 1:
            return fit_function(job.checkpoints[0])
        else:
            return tuple(fit_function(candidate) for candidate in job.checkpoints)

class TrainingService(object):
    def __init__(self, context : BaseContext, trainer : Trainer, evaluator : Evaluator, devices : List[str] = ['cpu'], n_jobs : int = 1, threading : bool = False, verbose : bool = False):
        super().__init__()
        self.context = context
        self.fitness_function = FitnessFunction(trainer=trainer, evaluator=evaluator)
        self.devices = devices
        self.n_jobs = n_jobs
        self.threading = threading
        self.verbose = verbose
        self.__pool = None
    
    def start(self):
        if self.__pool is not None:
            warnings.warn("Service is already running. Consider calling stop() when service is not in use.")
        self.__pool = ThreadPool(processes=self.n_jobs) if self.threading else self.context.Pool(processes=self.n_jobs)

    def stop(self):
        if self.__pool is None:
            warnings.warn("Service is not running.")
            return
        self.__pool.close()
        self.__pool.join()
        self.__pool = None

    def create_jobs(self, candidates : Sequence, step_size : int) -> Sequence[Job]:
        for candidate, device in zip(candidates, itertools.cycle(self.devices)):
            yield Job(candidate, step_size, device)

    def train(self, candidates : Sequence, step_size : int):
        jobs = self.create_jobs(candidates, step_size)
        for result in self.__pool.imap_unordered(self.fitness_function, jobs):
            yield result