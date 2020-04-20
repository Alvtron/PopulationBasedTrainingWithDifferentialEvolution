import os
import gc
import sys
import copy
import time
import math
import uuid
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
from .step import Step
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

class Trial:
    def __init__(self, return_queue, checkpoints : Tuple[Checkpoint], train_step_size : int, eval_step_size : int, train_shuffle : bool = False, eval_shuffle : bool = False):
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
        self.id_ = uuid.uuid4()
        self.return_queue = return_queue
        self.train_step_size = train_step_size
        self.eval_step_size = eval_step_size
        self.train_shuffle = train_shuffle
        self.eval_shuffle = eval_shuffle

STOP_FLAG = None

@dataclass
class FailMessage(object):
    sender_id : int
    text : str
    exception : str = None

class Worker(CONTEXT.Process):
    """A worker process that train and evaluate any available checkpoints provided from the train_queue. """
    def __init__(self, id : int, end_event, receive_queue, trainer, evaluator, tester = None, device : str = 'cpu', random_seed : int = 0, verbose : bool = False):
        super().__init__()
        self._id = id
        self.end_event = end_event
        self.receive_queue = receive_queue
        self.trainer = trainer
        self.evaluator = evaluator
        self.tester = tester
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

    def _process_trial(self, trial : Trial):
        if not trial.checkpoints:
            raise ValueError("No checkpoints available in trial-object.")
        step = Step(trainer=self.trainer, evaluator=self.evaluator, tester=self.tester,
            train_step_size=trial.train_step_size, eval_step_size=trial.eval_step_size,
            train_shuffle=trial.train_shuffle, eval_shuffle=trial.eval_shuffle,
            device=self.device, logger=self.__log)
        if isinstance(trial.checkpoints, Checkpoint):
            return step(trial.checkpoints)
        else:
            return tuple(step(checkpoint) for checkpoint in trial.checkpoints)

    def run(self):
        self.__log("running...")
        # set random state for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        while not self.end_event.is_set():
            # get next checkpoint from train queue
            self.__log("awaiting trial...")
            trial = self.receive_queue.get()
            if trial == STOP_FLAG:
                self.__log("STOP FLAG received. Stopping...")
                break
            if not isinstance(trial, Trial):
                self.__log("Received wrong trial-type.")
                raise TypeError('received wrong trial-type.')
            try:
                if self.cuda:
                    with torch.cuda.device(self.device):
                        result = self._process_trial(trial)
                else:
                    result = self._process_trial(trial)
                self.__log("returning trial result...")
                trial.return_queue.put(result)
            except Exception:
                import traceback
                self.__log("trial excecution failed! Exception:")
                traceback_stacktrace = traceback.format_exc()
                self.__log(str(traceback_stacktrace))
                fail_message = FailMessage(self._id, "trial excecution failed!", str(traceback_stacktrace))
                trial.return_queue.put(fail_message)
                # delete failed trial
                del trial
                break
            finally:
                # Explicitly trigger garbage collection to make sure that all
                # destructors are called...
                gc.collect()
        self.__log("stopped.")