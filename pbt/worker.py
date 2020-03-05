import os
import torch
import time
import random
import numpy as np
from dataclasses import dataclass

from .member import Checkpoint, MissingStateError
from .trainer import Trainer
from .evaluator import Evaluator

STOP_FLAG = None

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
mp = torch.multiprocessing.get_context('spawn')

@dataclass
class WorkerTask:
    checkpoint : Checkpoint
    trainer : Trainer
    evaluator : Evaluator
    step_size : int

@dataclass
class WorkerResponse:
    checkpoint : Checkpoint
    model_state : dict
    optimizer_state : dict

class Worker(mp.Process):
    """A worker process that train and evaluate any available checkpoints provided from the train_queue. """
    def __init__(self, id, end_event_global, evolve_queue, train_queue, random_seed : int = 0, verbose : bool = False):
        super().__init__()
        self.id = id
        self.end_event_global = end_event_global
        self.evolve_queue = evolve_queue
        self.train_queue = train_queue
        self.verbose = verbose
        # set random state for reproducibility
        random.seed(random_seed)
        numpy.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def __log(self, message : str, force_print : bool = False):
        prefix = f"Worker {self.id}, PID {os.getpid()}"
        full_message = f"{prefix}: {message}"
        if self.verbose or force_print:
            print(full_message)

    def run(self):
        self.__log("running...")
        while not self.end_event_global.is_set():
            self.__log("awaiting checkpoint...")
            # get next checkpoint from train queue
            task : WorkerTask = self.train_queue.get()
            # stop training loop if stop flag is received
            if task is STOP_FLAG:
                break
            checkpoint : Checkpoint = task.checkpoint
            # load checkpoint state
            self.__log("loading checkpoint state...")
            try:
                checkpoint.load_state(missing_ok=checkpoint.steps < task.step_size)
            except MissingStateError:
                self.__log(f"WARNING: received trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.", True)
            try:
                # train checkpoint model
                start_train_time_ns = time.time_ns()
                self.__log("training...")
                checkpoint.model_state, checkpoint.optimizer_state, checkpoint.epochs, checkpoint.steps, checkpoint.loss['train'] = task.trainer.train(
                    hyper_parameters=checkpoint.parameters,
                    model_state=checkpoint.model_state,
                    optimizer_state=checkpoint.optimizer_state,
                    epochs=checkpoint.epochs,
                    steps=checkpoint.steps,
                    step_size=task.step_size)
                checkpoint.time['train'] = float(time.time_ns() - start_train_time_ns) * float(10**(-9))
                # evaluate checkpoint model
                start_eval_time_ns = time.time_ns()
                self.__log("evaluating...")
                checkpoint.loss['eval'] = task.evaluator.eval(checkpoint.model_state)
                checkpoint.time['eval'] = float(time.time_ns() - start_eval_time_ns) * float(10**(-9))
                self.__log(f"Time: {checkpoint.time['train']:.2f}s train, {checkpoint.time['eval']:.2f}s eval")
                # unload checkpoint state
                self.__log("unloading checkpoint state...")
                checkpoint.unload_state()
                # send checkpoint back to controller
                self.evolve_queue.put(checkpoint)
                self.__log("checkpoint returned.")
            except Exception as e:
                self.__log(e)
                if task is not None:
                    self.__log(f"returning task with member {task.checkpoint.id} back to train queue...")
                    self.train_queue.put(task)
                break
            finally:
                self.__log("cleaning up GPU memory...")
                # ensure task deleted
                del task
                # release any unused GPU memory
                torch.cuda.empty_cache()
        self.__log("stopped.")