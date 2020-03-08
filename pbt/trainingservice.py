import os
import gc
import sys
import copy
import time
import math
import random
import warnings
import itertools
from functools import partial
from typing import List, Dict, Tuple, Sequence, Callable
from functools import partial
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.pool import ThreadPool

import torch
import numpy as np

import pbt.member
from .trainer import Trainer
from .evaluator import Evaluator
from .member import Checkpoint, MissingStateError

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

STOP_FLAG = None

def train_and_evaluate(checkpoint : Checkpoint, trainer : Trainer, evaluator : Evaluator, step_size : int, device : str, logger : Callable, verbose : bool = False):
    # load checkpoint state
    logger(f"loading state of checkpoint {checkpoint.id}...")
    try:
        checkpoint.load_state(device=device, missing_ok=checkpoint.steps < step_size)
    except MissingStateError:
        warnings.warn(f"WARNING on PID {os.getpid()}: trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.")
    # train checkpoint model
    logger(f"training checkpoint {checkpoint.id}...")
    trainer(checkpoint, step_size, device)
    # evaluate checkpoint model
    logger(f"evaluating checkpoint {checkpoint.id}...")
    evaluator(checkpoint, device)
    # unload checkpoint state
    logger(f"unloading state of checkpoint {checkpoint.id}...")
    checkpoint.unload_state()
    return checkpoint

class Job:
    def __init__(self, checkpoints : Tuple[Checkpoint], step_size : int):
        if isinstance(checkpoints, Sequence) and all(isinstance(checkpoint, Checkpoint) for checkpoint in checkpoints):
            self.checkpoints = checkpoints
        elif isinstance(checkpoints, Checkpoint):
            self.checkpoints = tuple([checkpoints])
        else:
            raise TypeError
        if not isinstance(step_size, int):
            raise TypeError
        self.step_size = step_size

class Worker(CONTEXT.Process):
    """A worker process that train and evaluate any available checkpoints provided from the train_queue. """
    def __init__(self, id, end_event, receive_queue, return_queue, trainer, evaluator, device : str = 'cpu', random_seed : int = 0, verbose : bool = False):
        super().__init__()
        self._id = id
        self.end_event = end_event
        self.receive_queue = receive_queue
        self.return_queue = return_queue
        self.trainer = trainer
        self.evaluator = evaluator
        self.cuda = device.startswith('cuda')
        self.device = torch.device(device)
        self.random_seed = random_seed
        self.verbose = verbose

    def __log(self, message : str):
        if not self.verbose:
            return
        prefix = f"PBT Worker {self._id} (PID {os.getpid()})"
        full_message = f"{prefix}: {message}"
        print(full_message)

    def process_job(self, job : Job):
        if not job.checkpoints:
            raise ValueError("No checkpoints available in job-object.")
        elif len(job.checkpoints) == 1:
            return train_and_evaluate(job.checkpoints[0], self.trainer, self.evaluator, job.step_size, self.device, self.__log, self.verbose)
        else:
            return tuple(train_and_evaluate(checkpoint, self.trainer, self.evaluator, job.step_size, self.device, self.__log, self.verbose) for checkpoint in job.checkpoints)

    def run(self):
        self.__log("running...")
        if self.cuda:
            torch.cuda.device(self.device)
        # set random state for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        while not self.end_event.is_set():
            # get next checkpoint from train queue
            self.__log("awaiting job...")
            job = self.receive_queue.get()
            if job is STOP_FLAG:
                self.__log("STOP FLAG received. Stopping...")
                break
            try:
                result = self.process_job(job)
                self.return_queue.put(result)
            except Exception as exception:
                self.__log("job excecution failed...")
                self.__log(str(exception))
                self.__log("returning task to send queue...")
                self.receive_queue.put(job)
                break
            finally:
                # Regular multiprocessing workers don't fully clean up after themselves,
                # so we have to explicitly trigger garbage collection to make sure that all
                # destructors are called...
                gc.collect()
        self.__log("stopped.")

class TrainingService(object):
    def __init__(self, trainer : Trainer, evaluator : Evaluator, devices : Sequence[str] = ('cpu',),
            n_jobs : int = 1, threading : bool = False, verbose : bool = False):
        super().__init__()
        if n_jobs < len(devices):
            raise ValueError("n_jobs must be larger or equal the number of devices.")
        self.context = torch.multiprocessing.get_context('spawn')
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._end_event = self.context.Event()
        self._return_queue = self.context.Queue()
        send_queues = [self.context.Queue() for _ in devices]
        workers = list()
        for id, send_queue, device in zip(range(self.n_jobs), itertools.cycle(send_queues), itertools.cycle(devices)):
            worker = Worker(id=id, end_event=self._end_event, receive_queue=send_queue, return_queue=self._return_queue,
                trainer=trainer, evaluator=evaluator, device = device, random_seed = id, verbose = verbose)
            workers.append(worker)
        self._workers : Tuple[Worker] = tuple(workers)

    def start(self):
        if self._workers is None:
            raise Exception("No workers found.")
        if any(worker.is_alive() for worker in self._workers):
            raise Exception("Service is already running. Consider calling stop() when service is not in use.")
        [worker.start() for worker in self._workers]

    def stop(self):
        if not any(worker.is_alive() for worker in self._workers):
            warnings.warn("Service is not running.")
            return
        self._end_event.set()
        [worker.receive_queue.put(STOP_FLAG) for worker in self._workers]
        [worker.join() for worker in self._workers]
        [worker.close() for worker in self._workers]

    def train(self, candidates : Sequence, step_size : int):
        n_sent = 0
        n_returned = 0
        for checkpoints, worker in zip(candidates, itertools.cycle(self._workers)):
            job = Job(checkpoints, step_size)
            worker.receive_queue.put(job)
            n_sent += 1
        while n_returned != n_sent:
            yield self._return_queue.get()
            n_returned += 1