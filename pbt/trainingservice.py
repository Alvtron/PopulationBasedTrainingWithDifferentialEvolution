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
from typing import List, Dict, Tuple, Sequence, Iterable, Callable, Union, Generator
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
from .worker import STOP_FLAG, FailMessage, InvalidInputMessage, Job, Worker
from .utils.cuda import get_gpu_memory_stats

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

class TrainingService(object):
    def __init__(self, trainer : Trainer, evaluator : Evaluator, devices : Sequence[str] = ('cpu',),
            n_jobs : int = 1, verbose : bool = False):
        super().__init__()
        if n_jobs < len(devices):
            raise ValueError("n_jobs must be larger or equal the number of devices.")
        self.verbose = verbose
        self._cuda = any(device.startswith('cuda') for device in devices)
        self._context = torch.multiprocessing.get_context('spawn')
        self._end_event = self._context.Event()
        self._return_queue = self._context.Queue()
        send_queues = [self._context.Queue() for _ in devices]
        workers = list()
        for id, send_queue, device in zip(range(n_jobs), itertools.cycle(send_queues), itertools.cycle(devices)):
            worker = Worker(id=id, end_event=self._end_event, receive_queue=send_queue, return_queue=self._return_queue,
                trainer=trainer, evaluator=evaluator, device = device, random_seed = id, verbose = verbose)
            workers.append(worker)
        self._workers : List[Worker] = list(workers)

    def _print(self, message : str) -> None:
        if not self.verbose:
            return
        full_message = f"Training Service: {message}"
        print(full_message)

    def _print_gpu_memory_stats(self) -> None:
        if not self.verbose or not self._cuda or os.name == 'nt':
            return
        memory_stats = get_gpu_memory_stats()
        memory_stats_formatted = (f"CUDA:{id} ({memory[0]}/{memory[1]}MB)" for id, memory in memory_stats.items())
        output = ', '.join(memory_stats_formatted)
        self._print(output)

    def _on_invalid_input_message(self, message : InvalidInputMessage):
        raise Exception(f"invalid input message received from worker {message.sender_id}: {message.text}.")

    def _on_fail_message(self, message : FailMessage) -> None:
        # print info
        self._print(f"fail message received from worker {message.sender_id}: {message.text}.")
        if message.exception:
            self._print(f"exception: {message.exception}.")

    def _respawn(self, worker_id : int) -> None:
        # stop existing worker
        self._print(f"terminating old worker with id {worker_id}...")
        worker = next((worker for worker in self._workers if worker.id == worker_id), None)
        if worker is None:
            raise KeyError(f"Could not find worker {worker_id}!")
        self._stop_worker(worker)
        # spawn new worker
        self._print(f"spawning new worker with id {worker.id}...")
        new_worker = Worker(id=worker.id, end_event=self._end_event, receive_queue=worker.receive_queue, return_queue=self._return_queue,
                trainer=worker.trainer, evaluator=worker.evaluator, device = worker.device, random_seed = worker.id, verbose = self.verbose)
        self._workers.append(new_worker)
        new_worker.start()

    def _stop_worker(self, worker : Worker) -> None:
        worker.terminate()
        time.sleep(1.0) # give worker one second to stop
        worker.close()
        self._workers.remove(worker)

    def start(self) -> None:
        if self._workers is None:
            raise Exception("no workers found.")
        if any(worker.is_alive() for worker in self._workers):
            raise Exception("service is already running. Consider calling stop() when service is not in use.")
        [worker.start() for worker in self._workers]

    def stop(self) -> None:
        if not any(worker.is_alive() for worker in self._workers):
            warnings.warn("service is not running.")
            return
        self._end_event.set()
        [worker.receive_queue.put(STOP_FLAG) for worker in self._workers]
        [worker.join() for worker in self._workers]
        [worker.close() for worker in self._workers]

    def train(self, candidates : Iterable[Union[Checkpoint, Tuple[Checkpoint,...]]], step_size : int) -> Generator[Union[Checkpoint, Tuple[Checkpoint,...]], None, None]:
        self._print(f"queuing candidates for training...")
        n_sent = 0
        for checkpoints, worker in zip(candidates, itertools.cycle(self._workers)):
            job = Job(checkpoints, step_size)
            worker.receive_queue.put(job)
            n_sent += 1
        self._print(f"receiving trained candidates...")
        n_returned = 0
        failed_workers = set()
        while n_returned != n_sent and len(failed_workers) < len(self._workers):
            result = self._return_queue.get()
            self._print_gpu_memory_stats()
            if isinstance(result, InvalidInputMessage):
                self._on_invalid_input_message(result)
            if isinstance(result, FailMessage):
                self._on_fail_message(result)
                failed_workers.add(result.sender_id)
                continue
            n_returned += 1
            yield result
        if len(failed_workers) == len(self._workers):
            raise Exception("All workers failed.")
        if n_returned < n_sent:
            if len(failed_workers) > 0:
                raise Exception(f"{len(failed_workers)} workers failed.")
            else:
                raise Exception(f"{n_sent - n_returned} candidates failed.")
        elif len(failed_workers) > 0:
            self._respawn(failed_workers)
        else:
            self._print("all candidates were trained successfully.")