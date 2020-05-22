import os
from abc import abstractmethod
from functools import partial
from warnings import warn
from typing import Callable, Union, Iterable, Sequence

from .trainer import Trainer
from .evaluator import Evaluator
from .member import Checkpoint, MissingStateError

def empty_logger(arg):
    pass

class DeviceCallable(object):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _print(self, message : str):
        if not self.verbose:
            return
        full_message = f"PID: {os.getpid()}, message: '{message}'"
        print(full_message)

    @abstractmethod
    def __call__(self, checkpoint: Checkpoint, device: str, **kwargs) -> Checkpoint:
        raise NotImplementedError()

class Step(DeviceCallable):
    def __init__(self, verbose: bool = False, **functions: Sequence[Union[Trainer, Evaluator]]):
        super().__init__(verbose)
        if not functions:
            raise ValueError('no functions provided.')
        for function_name, function in functions.items():
            if callable(function):
                continue
            raise ValueError(f"function '{function_name}' is not callable.")
        self.functions = functions

    def __call__(self, checkpoint: Checkpoint, device) -> Checkpoint:
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError(f"Received bad type: '{type(checkpoint).__name__}'. Expected: '{Checkpoint.__name__}'")
        # load checkpoint state
        self._print(f"loading state of checkpoint {checkpoint.id}...")
        try:
            checkpoint.load_state(device=device, missing_ok=checkpoint.steps == 0)
        except MissingStateError:
            warn(f"trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.")
        for index, (function_name, function) in enumerate(self.functions.items(), 1):
            # train checkpoint model
            self._print(f"({index}/{len(self.functions)}). called '{function_name}' with checkpoint {checkpoint.id}...")
            function(checkpoint, device=device)
        # unload checkpoint state
        self._print(f"unloading state of checkpoint {checkpoint.id}...")
        checkpoint.unload_state()
        return checkpoint