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
from abc import abstractmethod
from functools import partial
from typing import List, Dict, Tuple, Sequence, Callable
from functools import partial
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.pool import ThreadPool

import torch
import numpy as np

from pbt.device import DeviceCallable

# various settings for reproducibility
# set random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
# set torch settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
# multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
CONTEXT = torch.multiprocessing.get_context("spawn")

STOP_FLAG = None


class Trial:
    def __init__(self, return_queue, function: DeviceCallable, parameters: object):
        self.return_queue = return_queue
        self.function: DeviceCallable = function
        self.parameters = parameters

    def __call__(self, device: str):
        return self.function(self.parameters, device=device)


@dataclass
class FailMessage(object):
    sender_id: int
    text: str
    exception: str = None


class Worker(CONTEXT.Process):
    """A worker process that train and evaluate any available checkpoints provided from the train_queue. """

    def __init__(self, id: int, end_event, receive_queue, device: str = 'cpu', random_seed: int = 0, verbose: bool = False):
        super().__init__()
        self._id = id
        self.end_event = end_event
        self.receive_queue = receive_queue
        self.device = device
        self.random_seed = random_seed
        self.verbose = verbose

    @property
    def id(self):
        return self._id

    def __log(self, message: str):
        if not self.verbose:
            return
        prefix = f"Worker {self._id} (PID {os.getpid()})"
        full_message = f"{prefix}: {message}"
        print(full_message)

    def __process_trial(self, trial: Trial):
        if self.device.startswith('cuda'):
            with torch.cuda.device(self.device):
                return trial(self.device)
        else:
            return trial(self.device)

    def run(self):
        self.__log("running...")
        # set random state for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
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
                self.__log("running trial...")
                result = self.__process_trial(trial)
                self.__log("returning trial result...")
                trial.return_queue.put(result)
            except Exception:
                import traceback
                self.__log("trial excecution failed! Exception:")
                traceback_stacktrace = traceback.format_exc()
                self.__log(str(traceback_stacktrace))
                fail_message = FailMessage(
                    self._id, "trial excecution failed!", str(traceback_stacktrace))
                trial.return_queue.put(fail_message)
                # delete failed trial
                del trial
                break
            finally:
                # Explicitly trigger garbage collection to make sure that all destructors are called...
                gc.collect()
        self.__log("stopped.")
