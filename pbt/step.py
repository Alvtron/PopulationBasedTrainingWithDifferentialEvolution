from functools import partial
from warnings import warn
from typing import Callable

from .trainer import Trainer
from .evaluator import Evaluator
from .member import Checkpoint, MissingStateError

def empty_logger(arg):
    pass

class Step:
    def __init__(self, trainer : Trainer, evaluator : Evaluator, tester : Evaluator, train_step_size : int, eval_step_size : int = None,
            train_shuffle : bool = False, eval_shuffle : bool = False, device : str = 'cpu', logger : Callable[[str], None] = None):
        self.train = partial(trainer, step_size=train_step_size, shuffle=train_shuffle, device=device)
        self.eval = partial(evaluator, step_size=eval_step_size, shuffle=eval_shuffle, device=device)
        self.test = partial(tester, step_size=None, shuffle=False, device=device) if tester is not None else None
        self.device = device
        self.logger = logger if logger is not None else empty_logger

    def __call__(self, checkpoint : Checkpoint):
        # load checkpoint state
        self.logger(f"loading state of checkpoint {checkpoint.id}...")
        try:
            checkpoint.load_state(device=self.device, missing_ok=checkpoint.steps == 0)
        except MissingStateError:
            warn(f"trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.")
        # train checkpoint model
        self.logger(f"training checkpoint {checkpoint.id}...")
        self.train(checkpoint)
        # evaluate checkpoint model
        self.logger(f"evaluating checkpoint {checkpoint.id}...")
        self.eval(checkpoint)
        # test checkpoint model
        if self.test is not None:
            self.logger(f"testing checkpoint {checkpoint.id}...")
            self.test(checkpoint)
        # unload checkpoint state
        self.logger(f"unloading state of checkpoint {checkpoint.id}...")
        checkpoint.unload_state()
        return checkpoint