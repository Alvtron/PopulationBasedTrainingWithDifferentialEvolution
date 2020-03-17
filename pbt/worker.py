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
torch.multiprocessing.set_sharing_strategy('file_system')
CONTEXT = torch.multiprocessing.get_context("spawn")

def train_and_evaluate(checkpoint : Checkpoint, trainer : Trainer, evaluator : Evaluator, train_step_size : int, eval_step_size : int,
        device : str, logger : Callable, shuffle : bool = False, verbose : bool = False):
    # load checkpoint state
    logger(f"loading state of checkpoint {checkpoint.id}...")
    try:
        checkpoint.load_state(device=device, missing_ok=checkpoint.steps < train_step_size)
    except MissingStateError:
        warnings.warn(f"WARNING on PID {os.getpid()}: trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.")
    # train checkpoint model
    logger(f"training checkpoint {checkpoint.id}...")
    trainer(checkpoint, train_step_size, device, shuffle)
    # evaluate checkpoint model
    logger(f"evaluating checkpoint {checkpoint.id}...")
    evaluator(checkpoint, eval_step_size, device, shuffle)
    # unload checkpoint state
    logger(f"unloading state of checkpoint {checkpoint.id}...")
    checkpoint.unload_state()
    return checkpoint

class Job:
    def __init__(self, checkpoints : Tuple[Checkpoint], train_step_size : int, eval_step_size : int, shuffle : bool = False):
        if isinstance(checkpoints, Sequence) and all(isinstance(checkpoint, Checkpoint) for checkpoint in checkpoints):
            self.checkpoints = checkpoints
        elif isinstance(checkpoints, Checkpoint):
            self.checkpoints = checkpoints
        else:
            raise TypeError
        if not isinstance(train_step_size, int):
            raise TypeError
        if eval_step_size is not None and not isinstance(eval_step_size, int):
            raise TypeError
        self.train_step_size = train_step_size
        self.eval_step_size = eval_step_size
        self.shuffle = shuffle

STOP_FLAG = None

@dataclass
class InvalidInputMessage(object):
    sender_id : int
    text : str

@dataclass
class FailMessage(object):
    sender_id : int
    text : str
    exception : str = None

class Worker(CONTEXT.Process):
    """A worker process that train and evaluate any available checkpoints provided from the train_queue. """
    def __init__(self, id : int, end_event, receive_queue, return_queue, trainer, evaluator, device : str = 'cpu', random_seed : int = 0, verbose : bool = False):
        super().__init__()
        self._id = id
        self.end_event = end_event
        self.receive_queue = receive_queue
        self.return_queue = return_queue
        self.trainer = trainer
        self.evaluator = evaluator
        self.cuda = device.startswith('cuda')
        self.device = device
        self.random_seed = random_seed
        self.verbose = verbose

    @property
    def id(self):
        return self._id

    def __log(self, message : str):
        if not self.verbose:
            return
        prefix = f"Worker {self._id} (PID {os.getpid()})"
        full_message = f"{prefix}: {message}"
        print(full_message)

    def _process_job(self, job : Job):
        if not job.checkpoints:
            raise ValueError("No checkpoints available in job-object.")
        if isinstance(job.checkpoints, Checkpoint):
            return train_and_evaluate(checkpoint=job.checkpoints, trainer=self.trainer, evaluator=self.evaluator,
                train_step_size=job.train_step_size, eval_step_size=job.eval_step_size,
                device=self.device, logger=self.__log, shuffle=job.shuffle, verbose=self.verbose)
        else:
            return tuple(train_and_evaluate(checkpoint=checkpoint, trainer=self.trainer, evaluator=self.evaluator,
                train_step_size=job.train_step_size, eval_step_size=job.eval_step_size,
                device=self.device, logger=self.__log, shuffle=job.shuffle, verbose=self.verbose)
                for checkpoint in job.checkpoints)

    def run(self):
        self.__log("running...")
        # set random state for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        while not self.end_event.is_set():
            # get next checkpoint from train queue
            self.__log("awaiting job...")
            job = self.receive_queue.get()
            if job == STOP_FLAG:
                self.__log("STOP FLAG received. Stopping...")
                break
            if not isinstance(job, Job):
                self.__log("Received wrong job-type.")
                self.return_queue.put(InvalidInputMessage(self._id, f"Wrong job-type received: {job}!"))
                continue
            try:
                if self.cuda:
                    with torch.cuda.device(self.device):
                        result = self._process_job(job)
                else:
                    result = self._process_job(job)
                self.__log("returning job result...")
                self.return_queue.put(result)
            except Exception:
                import traceback
                self.__log("job excecution failed! Exception:")
                traceback_stacktrace = traceback.format_exc()
                self.__log(str(traceback_stacktrace))
                fail_message = FailMessage(self._id, "Job excecution failed!", str(traceback_stacktrace))
                self.return_queue.put(fail_message)
                # delete failed job
                del job
                break
            finally:
                # Explicitly trigger garbage collection to make sure that all
                # destructors are called...
                gc.collect()
        self.__log("stopped.")