from abc import ABC
from functools import partial
from warnings import warn
from typing import Callable, Union, Iterable

from .trainer import Trainer
from .evaluator import Evaluator
from .member import Checkpoint, MissingStateError

def empty_logger(arg):
    pass

class DeviceCallable(ABC):
    def __call__(self, device: str, logger: Callable[[str], None]):
        pass

class Step(DeviceCallable):
    def __init__(self, trainer: Trainer, evaluator: Evaluator, train_step_size: int, eval_step_size: int = None,
            train_shuffle: bool = False, eval_shuffle: bool = False, tester: Evaluator = None):
        self.train = partial(trainer, step_size=train_step_size, shuffle=train_shuffle)
        self.eval = partial(evaluator, step_size=eval_step_size, shuffle=eval_shuffle)
        self.test = partial(tester, step_size=None, shuffle=False) if tester is not None else None

    def __process(self, checkpoint: Checkpoint, device: str, logger: Callable[[str], None]):
        # correct logger
        if logger is None:
            logger = empty_logger
        # load checkpoint state
        logger(f"loading state of checkpoint {checkpoint.id}...")
        try:
            checkpoint.load_state(device=device, missing_ok=checkpoint.steps == 0)
        except MissingStateError:
            warn(f"trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.")
        # train checkpoint model
        logger(f"training checkpoint {checkpoint.id}...")
        self.train(checkpoint, device=device)
        # evaluate checkpoint model
        logger(f"evaluating checkpoint {checkpoint.id}...")
        self.eval(checkpoint, device=device)
        # test checkpoint model
        if self.test is not None:
            logger(f"testing checkpoint {checkpoint.id}...")
            self.test(checkpoint, device=device)
        # unload checkpoint state
        logger(f"unloading state of checkpoint {checkpoint.id}...")
        checkpoint.unload_state()
        return checkpoint

    def __call__(self, checkpoints: Union[Checkpoint, Iterable[Checkpoint]], device: str = "cpu", logger: Callable[[str], None] = None):
        if not checkpoints:
            raise ValueError("No checkpoints available.")
        if isinstance(checkpoints, Checkpoint):
            # when single checkpoint is given, return a single processed checkpoint back
            return self.__process(checkpoint=checkpoints, device=device, logger=logger)
        else:
            # when multiple checkpoints are given, return a tuple of processed checkpoints back
            return tuple(self.__process(checkpoint=checkpoint, device=device, logger=logger) for checkpoint in checkpoints)