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
from typing import List, Dict, Tuple, Sequence, Callable, Union, Generator, Any
from functools import partial
from dataclasses import dataclass
from multiprocessing.managers import EventProxy
from multiprocessing.queues import Queue
from multiprocessing.pool import ThreadPool

import torch
import numpy as np

from pbt.device import DeviceCallable

# set torch multiprocessing settings
torch.multiprocessing.set_sharing_strategy('file_system')
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

STOP_FLAG = None


def map_to_threads(function, parameters):
    with ThreadPool(processes=len(parameters)) as pool:
        yield from pool.imap_unordered(function, parameters)


class ThreadTask:
    def __init__(self, return_queue, function: DeviceCallable, parameters):
        if return_queue is None:
            raise TypeError(f"the 'return_queue' specified was None.")
        if not isinstance(function, DeviceCallable):
            raise TypeError(f"the 'function' specified was of wrong type {type(function)}, expected {DeviceCallable}.")
        if parameters is not None and not isinstance(parameters, (list, tuple)):
            raise TypeError(f"the 'parameters' specified was of wrong type {type(parameters)}, expected {list} or {tuple}.")
        self.return_queue = return_queue
        self.function: DeviceCallable = function
        self.parameters = parameters

    def run(self, device: str) -> Generator[Any, None, None]:
        device_function = partial(self.function, device=device)
        if not device.startswith('cuda'):
            yield from map_to_threads(device_function, self.parameters)
        with torch.cuda.device(device):
            yield from map_to_threads(device_function, self.parameters)

@dataclass
class FailMessage:
    sender_id: int
    text: str
    exception: str = None


class DeviceWorker(torch.multiprocessing.Process):
    """A worker process that train and evaluate any available checkpoints provided from the train_queue. """

    def __init__(self, uid: Union[int, str], end_event: EventProxy, receive_queue: Queue, device: str = 'cpu', random_seed: int = 0, verbose: bool = False):
        super().__init__()
        if not isinstance(uid, (int, str)):
            raise TypeError(f"the 'uid' specified was of wrong type {type(uid)}, expected {str} or {int}.")
        if not isinstance(end_event, EventProxy):
            raise TypeError(f"the 'end_event' specified was of wrong type {type(end_event)}, expected {EventProxy}.")
        if not isinstance(receive_queue, Queue):
            raise TypeError(f"the 'receive_queue' specified was of wrong type {type(receive_queue)}, expected {Queue}.")
        if not isinstance(device, str):
            raise TypeError(f"the 'device' specified was of wrong type {type(device)}, expected {str}.")
        if not isinstance(random_seed, int):
            raise TypeError(f"the 'random_seed' specified was of wrong type {type(random_seed)}, expected {int}.")
        if not isinstance(verbose, bool):
            raise TypeError(f"the 'verbosity' specified was of wrong type {type(verbose)}, expected {bool}.", )
        self.uid = uid
        self.end_event = end_event
        self.receive_queue = receive_queue
        self.device = device
        self.random_seed = random_seed
        self.verbose = verbose

    def __log(self, message: str):
        if not self.verbose:
            return
        prefix = f"Worker {self.uid} (PID {os.getpid()})"
        full_message = f"{prefix}: {message}"
        print(full_message)

    def run(self):
        self.__log("running...")
        # set random state for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        # create threadpool
        while not self.end_event.is_set():
            # get next checkpoint from train queue
            self.__log("awaiting task...")
            task = self.receive_queue.get()
            if task == STOP_FLAG:
                self.__log("STOP FLAG received. Stopping...")
                break
            if not isinstance(task, ThreadTask):
                self.__log("Received wrong task-type.")
                raise TypeError(f"the 'task' received was of wrong type {type(task)}, expected {ThreadTask}.", )
            try:
                self.__log("running task...")
                results = task.run(self.device)
                for task_nr, result in enumerate(results, 1):
                    self.__log(f"returning task result {task_nr} of {len(task.parameters)}...")
                    task.return_queue.put(result)
            except Exception:
                import traceback
                self.__log("task excecution failed! Exception:")
                traceback_stacktrace = traceback.format_exc()
                self.__log(str(traceback_stacktrace))
                fail_message = FailMessage(
                    self.uid, "task excecution failed!", str(traceback_stacktrace))
                task.return_queue.put(fail_message)
                # delete failed task
                del task
                break
            finally:
                # Explicitly trigger garbage collection to make sure that all destructors are called...
                gc.collect()
        self.__log("stopped.")
