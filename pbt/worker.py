import os
import torch
import time
import random
import warnings
import copy
from dataclasses import dataclass
from typing import Tuple, Sequence, Callable
from functools import partial

import numpy as np

from .member import Checkpoint, MissingStateError
from .trainer import Trainer
from .evaluator import Evaluator

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
mp = torch.multiprocessing.get_context('spawn')

STOP_FLAG = None

def train_and_evaluate(checkpoint : Checkpoint, trainer : Trainer, evaluator : Evaluator, step_size : int, device : str, logger : Callable, verbose : bool = False):
    # load checkpoint state
    logger("loading checkpoint state...")
    try:
        checkpoint.load_state(device=device, missing_ok=checkpoint.steps < step_size)
    except MissingStateError:
        warnings.warn(f"WARNING on PID {os.getpid()}: trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.")
    # train checkpoint model
    logger("training...")
    trainer(checkpoint, step_size, device)
    # evaluate checkpoint model
    logger("evaluating...")
    evaluator(checkpoint, device)
    # unload checkpoint state
    logger("unloading checkpoint state...")
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

class Worker(mp.Process):
    """A worker process that train and evaluate any available checkpoints provided from the train_queue. """
    def __init__(self, id, end_event, receive_queue, return_queue, trainer, evaluator, device : str = 'cpu', random_seed : int = 0, verbose : bool = False):
        super().__init__()
        self._id = id
        self.end_event = end_event
        self.receive_queue = receive_queue
        self.return_queue = return_queue
        self.trainer = trainer
        self.evaluator = evaluator
        self.device = device
        self.verbose = verbose
        # set random state for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

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
        self.__log("stopped.")