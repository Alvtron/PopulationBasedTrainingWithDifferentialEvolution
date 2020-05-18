import os
from abc import ABC
from functools import partial
from warnings import warn
from typing import Callable, Union, Iterable, Sequence

from .trainer import Trainer
from .evaluator import Evaluator
from .member import Checkpoint, MissingStateError

def empty_logger(arg):
    pass

class DeviceCallable(ABC):
    def __call__(self, device: str, **kwargs):
        raise NotImplementedError()

class Step(DeviceCallable):
    def __init__(self, verbose: bool = False, **functions: Sequence[Union[Trainer, Evaluator]]):
        if not functions:
            raise ValueError('no functions provided.')
        
        self.functions = functions
        self.verbose = verbose

    def __print(self, message : str):
        if not self.verbose:
            return
        full_message = f"PID: {os.getpid()}, message: '{message}'"
        print(full_message)

    def __call__(self, checkpoint: Checkpoint, device):
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError(f"Received bad type: '{type(checkpoint).__name__}'. Expected: '{Checkpoint.__name__}'")
        # copy checkpoint
        checkpoint = checkpoint.copy()
        # load checkpoint state
        self.__print(f"loading state of checkpoint {checkpoint.id}...")
        try:
            checkpoint.load_state(device=device, missing_ok=checkpoint.steps == 0)
        except MissingStateError:
            warn(f"trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.")
        for index, (function_name, function) in enumerate(self.functions.items()):
            # train checkpoint model
            self.__print(f"{index}. performing '{function_name}' on checkpoint {checkpoint.id}...")
            checkpoint = function(checkpoint, device=device)
        # unload checkpoint state
        self.__print(f"unloading state of checkpoint {checkpoint.id}...")
        checkpoint.unload_state()
        return checkpoint