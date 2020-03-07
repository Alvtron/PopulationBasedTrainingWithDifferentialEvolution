import os
import sys
import time
import math
import random
import itertools
import warnings
import numpy as np
from typing import List, Dict, Tuple, Sequence, Callable
from functools import partial
from multiprocessing.context import BaseContext
from multiprocessing.pool import ThreadPool

import torch

import pbt.member
from .trainer import Trainer
from .evaluator import Evaluator
from .worker import Worker, Job, STOP_FLAG
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

class TrainingService(object):
    def __init__(self, trainer : Trainer, evaluator : Evaluator, devices : Sequence[str] = ('cpu',),
            n_jobs : int = 1, threading : bool = False, verbose : bool = False):
        super().__init__()
        if n_jobs < len(devices):
            raise ValueError("n_jobs must be larger or equal the number of devices.")
        self.context = torch.multiprocessing.get_context('spawn')
        self.trainer = trainer
        self.evaluator = evaluator
        self.devices = tuple(devices)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._workers : Tuple[Worker] = None
        self._return_queue = self.context.Queue()

    def _create_processes(self):
        self._end_event = self.context.Event()
        send_queues = [self.context.Queue() for _ in self.devices]
        for id, send_queue, device in zip(range(self.n_jobs), itertools.cycle(send_queues), itertools.cycle(self.devices)):
            worker = Worker(id=id, end_event=self._end_event, receive_queue=send_queue, return_queue=self._return_queue,
                trainer=self.trainer, evaluator=self.evaluator, device = device, random_seed = id, verbose = self.verbose)
            print(id, send_queue, device)
            worker.start()
            yield worker

    def start(self):
        if self._workers is not None:
            raise Exception("Service is already running. Consider calling stop() when service is not in use.")
        self._workers = tuple(self._create_processes())

    def stop(self):
        if not self._workers:
            warnings.warn("Service is not running.")
            return
        self._end_event.set()
        [worker.receive_queue.put(STOP_FLAG) for worker in self._workers]
        [worker.join() for worker in self._workers]
        [worker.close() for worker in self._workers]
        self._workers = None

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