import os
import gc
import sys
import random
from functools import partial
from typing import Callable, Union, Sequence, Generator, Any
from dataclasses import dataclass
from threading import Thread
from multiprocessing.managers import EventProxy
from multiprocessing.queues import Queue
from multiprocessing.pool import ThreadPool

import torch
import numpy as np

from pbt.device import set_global_device, initialize_cuda_device

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


def map_to_threads(function: Callable[..., Any], parameters: Sequence[Any], n_threads: int = None):
    n_threads = len(parameters) if n_threads is None else n_threads
    with ThreadPool(processes=n_threads) as pool:
        yield from pool.imap_unordered(function, parameters)


@dataclass
class FailMessage:
    sender_id: int
    text: str
    exception: str = None


class AsyncThreadTask:
    def __init__(self, return_queue, function: Callable[..., Any], parameters: Sequence[Any]):
        if return_queue is None:
            raise TypeError(f"the 'return_queue' specified was None.")
        if not callable(function):
            raise TypeError(f"the 'function' specified was not callable.")
        if not isinstance(parameters, (list, tuple)):
            raise TypeError(f"the 'parameters' specified was of wrong type {type(parameters)}, expected {list} or {tuple}.")
        self.return_queue = return_queue
        self.function = function
        self.parameters = parameters

    def run(self) -> Generator[Any, None, None]:
        yield from map_to_threads(self.function, self.parameters)


class DeviceWorker(torch.multiprocessing.Process):
    """A worker process that train and evaluate any available checkpoints provided from the receive_queue. """
    def __init__(self, uid: Union[int, str], end_event: EventProxy, receive_queue: Queue, device: str, random_seed: int = 0, verbose: bool = False):
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
        self.random_seed = random_seed
        self.device = device
        self.verbose = verbose
        # initialize CUDA if device is a GPU
        if device.startswith('cuda'):
            initialize_cuda_device(device)

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
        # set global processing device
        set_global_device(self.device)
        # start worker rutine
        while not self.end_event.is_set():
            # get next checkpoint from train queue
            self.__log("awaiting task...")
            task = self.receive_queue.get()
            if task == STOP_FLAG:
                self.__log("STOP FLAG received. Stopping...")
                break
            if not isinstance(task, AsyncThreadTask):
                self.__log("received wrong task-type.")
                raise TypeError(f"the 'task' received was of wrong type {type(task)}, expected {AsyncThreadTask}.", )
            try:
                # run tasks and return results each by each
                # the moment they are ready
                [task.return_queue.put(result) for result in task.run()]
            except Exception:
                import traceback
                self.__log("task excecution failed! Exception:")
                traceback_stacktrace = traceback.format_exc()
                self.__log(str(traceback_stacktrace))
                fail_message = FailMessage(self.uid, "task excecution failed!", str(traceback_stacktrace))
                task.return_queue.put(fail_message)
                # delete failed task
                del task
                break
            finally:
                # Explicitly trigger garbage collection to make
                # sure that all destructors are called
                gc.collect()
        self.__log("stopped.")