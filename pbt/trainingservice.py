import os
import gc
import sys
import time
import math
import random
import itertools
import warnings
import subprocess
from typing import List, Dict, Tuple, Sequence, Callable
from functools import partial
from multiprocessing.context import BaseContext
from multiprocessing.pool import ThreadPool

import numpy as np
import torch

import pbt.member
from .trainer import Trainer
from .evaluator import Evaluator
from .member import Checkpoint, MissingStateError
from .utils.iterable import split_number_evenly
from .utils.cuda import get_gpu_memory_map

# various settings for reproducibility
# set random state 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# set torch settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
# multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')
CONTEXT = torch.multiprocessing.get_context("spawn")

def log(message : str):
    prefix = f"PID {os.getpid()}"
    print(f"{prefix}: {message}")

def train_and_evaluate(checkpoint : Checkpoint, trainer : Trainer, evaluator : Evaluator, step_size : int, device, verbose : bool = False):
    # load checkpoint state
    if verbose: log("loading checkpoint state...")
    try:
        checkpoint.load_state(device=device, missing_ok=checkpoint.steps < step_size)
    except MissingStateError:
        warnings.warn(f"WARNING on PID {os.getpid()}: trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.")
    if verbose: log(f"Memory allocated on device {device}: {torch.cuda.memory_allocated(device)}")
    # train checkpoint model
    if verbose: log("training...")
    trainer(checkpoint, step_size, device)
    # evaluate checkpoint model
    if verbose: log("evaluating...")
    evaluator(checkpoint, device)
    # unload checkpoint state
    if verbose: log("unloading checkpoint state...")
    checkpoint.unload_state()
    return checkpoint

class Job:
    def __init__(self, checkpoints : Tuple[Checkpoint], step_size : int, device : str, verbose : bool = False):
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
        self.device = torch.device(device)
        self.verbose = verbose

class FitnessFunction(object):
    def __init__(self, trainer : Trainer, evaluator : Evaluator):
        self.trainer = trainer
        self.evaluator = evaluator

    def __call__(self, job : Job) -> Tuple[Checkpoint, ...]:
        fit_function = partial(train_and_evaluate, trainer=self.trainer, evaluator=self.evaluator,
            step_size=job.step_size, device=job.device, verbose=job.verbose)
        if not job.checkpoints:
            raise ValueError("No checkpoints available in job-object.")
        elif len(job.checkpoints) == 1:
            return fit_function(job.checkpoints[0])
        else:
            return tuple(fit_function(candidate) for candidate in job.checkpoints)

class TrainingService(object):
    def __init__(self, trainer : Trainer, evaluator : Evaluator,
            devices : List[str] = ['cpu'], n_jobs : int = 1, threading : bool = False, verbose : bool = False):
        super().__init__()
        self.fitness_function = FitnessFunction(trainer=trainer, evaluator=evaluator)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.devices = tuple(devices)
        self.__pools = None
    
    def is_alive(self):
        return self.__pools is not None

    def start(self):
        if self.is_alive():
            raise Exception("Service is already running. Consider calling stop() when service is not in use.")
        n_job_distribution = split_number_evenly(self.n_jobs, len(self.devices))
        self.__pools = tuple(CONTEXT.Pool(processes=n_jobs) for device, n_jobs in zip(self.devices, n_job_distribution))

    def stop(self):
        if not self.is_alive():
            warnings.warn("Service is not running.")
            return
        [pool.close() for pool in self.__pools]
        [pool.join() for pool in self.__pools]
        self.__pools = None
    
    def terminate(self):
        if not self.is_alive():
            warnings.warn("Service is not running.")
            return
        [pool.terminate() for pool in self.__pools]
        [pool.join() for pool in self.__pools]
        self.__pools = None

    def create_jobs(self, candidates : Sequence, step_size : int) -> Sequence[Job]:
        for checkpoints, device in zip(candidates, itertools.cycle(self.devices)):
            yield Job(checkpoints, step_size, device, self.verbose)

    def train(self, candidates : Sequence, step_size : int):
        jobs = self.create_jobs(candidates, step_size)
        tasks = [pool.apply_async(self.fitness_function, (job,)) for job, pool in zip(jobs, itertools.cycle(self.__pools))]
        time.sleep(100)
        for task in tasks:
            result = task.get()
            gpu_memory_usage = get_gpu_memory_map()
            print(gpu_memory_usage)
            print(result.__dict__)
            yield result