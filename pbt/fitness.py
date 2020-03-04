import os
import torch
import time
import random
import numpy
from dataclasses import dataclass
from typing import Callable, Tuple
from functools import partial

from .evolution import EvolveEngine
from .member import Checkpoint, Generation, MissingStateError
from .trainer import Trainer
from .evaluator import Evaluator

# reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
# multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')

VERBOSE = False

def log(message : str, force_print : bool = False):
    prefix = f"PID {os.getpid()}"
    full_message = f"{prefix}: {message}"
    if VERBOSE: print(full_message)

def train(checkpoint : Checkpoint, trainer : Trainer, step_size : int, device : str):
    log("training...")
    start_train_time_ns = time.time_ns()
    checkpoint.model_state, checkpoint.optimizer_state, checkpoint.epochs, checkpoint.steps, checkpoint.loss['train'] = trainer(
        hyper_parameters=checkpoint.hyper_parameters,
        model_state=checkpoint.model_state,
        optimizer_state=checkpoint.optimizer_state,
        epochs=checkpoint.epochs,
        steps=checkpoint.steps,
        step_size=step_size,
        device=device)
    checkpoint.time['train'] = float(time.time_ns() - start_train_time_ns) * float(10**(-9))
    return checkpoint

def evaluate(checkpoint : Checkpoint, evaluator : Evaluator, device : str):
    start_eval_time_ns = time.time_ns()
    log("evaluating...")
    checkpoint.loss['eval'] = evaluator(checkpoint.model_state, device)
    checkpoint.time['eval'] = float(time.time_ns() - start_eval_time_ns) * float(10**(-9))
    return checkpoint

def train_and_evaluate(checkpoint : Checkpoint, trainer : Trainer, evaluator : Evaluator, step_size : int, device : str):
        # load checkpoint state
        log("loading checkpoint state...")
        try:
            checkpoint.load_state(device=device, missing_ok=checkpoint.steps < step_size)
        except MissingStateError:
            log(f"WARNING: received trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.", True)
        # train checkpoint model
        checkpoint = train(checkpoint, trainer, step_size, device)
        # evaluate checkpoint model
        checkpoint = evaluate(checkpoint, evaluator, device)
        # unload checkpoint state
        log("unloading checkpoint state...")
        checkpoint.unload_state()
        return checkpoint