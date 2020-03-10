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
from .worker import STOP_FLAG, Job, Worker
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
torch.multiprocessing.set_sharing_strategy('file_descriptor')

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
        self._return_queue = self._context.SimpleQueue()
        send_queues = [self._context.SimpleQueue() for _ in devices]
        workers = list()
        for id, send_queue, device in zip(range(n_jobs), itertools.cycle(send_queues), itertools.cycle(devices)):
            worker = Worker(id=id, end_event=self._end_event, receive_queue=send_queue, return_queue=self._return_queue,
                trainer=trainer, evaluator=evaluator, device = device, random_seed = id, verbose = verbose)
            workers.append(worker)
        self._workers : Tuple[Worker] = tuple(workers)

    def _print_gpu_memory_stats(self):
        if not self.verbose or not self._cuda:
            return
        memory_stats = get_gpu_memory_stats()
        memory_stats_formatted = (f"CUDA:{id} ({memory[0]}/{memory[1]}MB)" for id, memory in memory_stats.items())
        output = ', '.join(memory_stats_formatted)
        print(output)

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
            self._print_gpu_memory_stats()
            n_returned += 1