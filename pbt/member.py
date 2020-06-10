from __future__ import annotations
import gc
import copy
import math
import shutil
import warnings
from pathlib import Path
from abc import abstractmethod
from collections import defaultdict, deque
from itertools import islice, chain
from datetime import datetime
from typing import List, Dict, Iterable, Sequence, Generator, Iterator

import torch

from .utils.conversion import dict_to_binary, binary_to_dict
from .utils.iterable import chunks, insert_sequence
from .hyperparameters import ContiniousHyperparameter, _Hyperparameter, Hyperparameters
from .utils.iterable import modify_iterable


def prepare_score(value):
    if value is None:
        return None
    if isinstance(value, (float, int)):
        if math.isfinite(value):
            return value
        else:
            return None
    if hasattr(value, "eval_score"):
        return prepare_score(value.eval_score())
    raise TypeError(f"type {type(value)} is not supported.")


class MissingStateError(Exception):
    pass


class Checkpoint(object):
    '''Class for keeping track of a checkpoint, where the model- and optimizer state are stored in system memory.'''
    TRAIN_LABEL = 'train'
    EVAL_LABEL = 'eval'
    TEST_LABEL = 'test'
    MODEL_STATE_PROPERTY = 'model_state'
    OPTIMIZER_STATE_PROPERTY = 'optimizer_state'

    def __init__(self, uid: object, parameters: Hyperparameters, loss_metric: str, eval_metric: str, minimize: bool):
        if uid is None:
            raise TypeError(f"the 'uid' specified was None.")
        if not isinstance(parameters, Hyperparameters):
            raise TypeError(f"the 'parameters' specified was of wrong type {type(parameters)}, expected {Hyperparameters}.")
        if not isinstance(loss_metric, str):
            raise TypeError(f"the 'loss_metric' specified was of wrong type {type(loss_metric)}, expected {str}.")
        if not isinstance(eval_metric, str):
            raise TypeError(f"the 'eval_metric' specified was of wrong type {type(eval_metric)}, expected {str}.")
        if not isinstance(minimize, bool):
            raise TypeError(f"the 'minimize' specified was of wrong type {type(minimize)}, expected {bool}.")
        # properties
        self.uid: object = uid
        self.parameters: Hyperparameters = parameters
        self.model_state: dict = None
        self.optimizer_state: dict = None
        self.loss_metric: str = loss_metric
        self.eval_metric: str = eval_metric
        self.minimize: bool = minimize
        self.loss: Dict[str, dict] = dict()
        self.time: Dict[str, float] = dict()
        self.steps: int = 0
        self.epochs: int = 0

    def __str__(self) -> str:
        return f"member {self.uid}, step {self.steps}, epoch {self.epochs}"

    def __getitem__(self, index: int) -> float:
        return self.parameters[index].normalized

    def __setitem__(self, index: int, value: float) -> None:
        if not isinstance(value, float):
            raise TypeError()
        self.parameters[index].normalized = value

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.uid == other.uid and self.steps == other.steps

    def __ne__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return True
        return self.uid != other.uid or self.steps != other.steps

    def __lt__(self, other) -> bool:
        x = prepare_score(self)
        y = prepare_score(other)
        if x is None and y is None:
            return False
        elif x is None:
            return True
        elif y is None:
            return False
        return x < y if not self.minimize else x > y

    def __le__(self, other) -> bool:
        x = prepare_score(self)
        y = prepare_score(other)
        if x is None and y is None:
            return True
        elif x is None:
            return True
        elif y is None:
            return False
        return x <= y if not self.minimize else x >= y

    def __gt__(self, other) -> bool:
        x = prepare_score(self)
        y = prepare_score(other)
        if x is None and y is None:
            return False
        elif x is None:
            return False
        elif y is None:
            return True
        return x > y if not self.minimize else x < y

    def __ge__(self, other) -> bool:
        x = prepare_score(self)
        y = prepare_score(other)
        if x is None and y is None:
            return True
        elif x is None:
            return False
        elif y is None:
            return True
        return x >= y if not self.minimize else x <= y

    def train_score(self) -> float:
        if Checkpoint.TRAIN_LABEL not in self.loss or self.eval_metric not in self.loss[Checkpoint.TRAIN_LABEL]:
            return None
        return self.loss[Checkpoint.TRAIN_LABEL][self.eval_metric]

    def eval_score(self) -> float:
        if Checkpoint.EVAL_LABEL not in self.loss or self.eval_metric not in self.loss[Checkpoint.EVAL_LABEL]:
            return None
        return self.loss[Checkpoint.EVAL_LABEL][self.eval_metric]

    def test_score(self) -> float:
        if Checkpoint.TEST_LABEL not in self.loss or self.eval_metric not in self.loss[Checkpoint.TEST_LABEL]:
            return None
        return self.loss[Checkpoint.TEST_LABEL][self.eval_metric]

    def copy_score(self, other) -> None:
        """Replace own score with score from the other member."""
        self.loss = copy.deepcopy(other.loss)

    def has_model_state(self) -> bool:
        """Returns whether the model state exists within this instance."""
        return hasattr(self, Checkpoint.MODEL_STATE_PROPERTY) and self.model_state is not None

    def has_optimizer_state(self) -> bool:
        """Returns whether the optimizer state exists within this instance."""
        return hasattr(self, Checkpoint.OPTIMIZER_STATE_PROPERTY) and self.optimizer_state is not None

    def has_state(self) -> bool:
        """Returns whether the model- and optimizer state exist within this instance."""
        return self.has_model_state() and self.has_optimizer_state()

    def load_state(self, device: str = 'cpu', missing_ok: bool = False) -> None:
        """Move all tensors in state to specified device. Call unload_state to move state back to cpu. Raises error if states are not available."""
        # load model state
        if self.has_model_state():
            modify_iterable(self.model_state, lambda x: x.to(
                device), lambda x: isinstance(x, torch.Tensor))
        elif missing_ok:
            self.model_state = None
        else:
            raise MissingStateError(f"Model state is missing.")
        # load optimizer state
        if self.has_optimizer_state():
            modify_iterable(self.optimizer_state, lambda x: x.to(
                device), lambda x: isinstance(x, torch.Tensor))
        elif missing_ok:
            self.optimizer_state = None
        else:
            raise MissingStateError(f"Optimizer state is missing.")

    def unload_state(self, missing_ok: bool = False) -> None:
        """Move all tensors in state to cpu. Call load_state to load state on a specific device. Raises error if states are not available."""
        target_device = 'cpu'
        # unload model state
        if self.has_model_state():
            modify_iterable(self.model_state, lambda x: x.to(
            target_device), lambda x: isinstance(x, torch.Tensor))
        elif missing_ok:
            self.optimizer_state = None
        else:
            raise MissingStateError("Attempted to unload model when model state is None.")
        # unload optimizer state
        if self.has_optimizer_state():
            modify_iterable(self.optimizer_state, lambda x: x.to(
            target_device), lambda x: isinstance(x, torch.Tensor))
        elif missing_ok:
            self.optimizer_state = None
        else:
            raise MissingStateError("Attempted to unload optimizer when optimizer state is None.")

    def delete_state(self) -> None:
        """Deletes the states from memory."""
        # delete model state from memory
        if hasattr(self, Checkpoint.MODEL_STATE_PROPERTY):
            del self.model_state
        # delete optimizer state from memory
        if hasattr(self, Checkpoint.OPTIMIZER_STATE_PROPERTY):
            del self.optimizer_state

    def copy_state(self, other: Checkpoint, warn_if_missing: bool = True) -> None:
        """Replace own model state and optimizer state with the ones from the other checkpoint."""
        if not isinstance(other, self.__class__):
            raise TypeError(f"the 'other' specified was of wrong type {type(other)}, expected {self.__class__}.")
        if self == other:
            warnings.warn("Attempted to copy states between the same object.")
            return
        # copy model state
        if not other.has_model_state():
            if warn_if_missing:
                warnings.warn("Copying empty model state")
            self.model_state = None
        else:
            self.model_state = copy.deepcopy(other.model_state)
        # copy optimizer state
        if not other.has_optimizer_state():
            if warn_if_missing:
                warnings.warn("Copying empty optimizer state")
            self.optimizer_state = None
        else:
            self.optimizer_state = copy.deepcopy(other.optimizer_state)

    def copy_parameters(self, other: Checkpoint) -> None:
        """Replace own hyper-parameters with the ones from the other checkpoint."""
        if not isinstance(other, self.__class__):
            raise TypeError(f"the 'other' specified was of wrong type {type(other)}, expected {self.__class__}.")
        self.parameters = copy.deepcopy(other.parameters)

    def copy(self) -> Checkpoint:
        return copy.deepcopy(self)

    def register_time(self, tag: str, start: datetime, end: datetime) -> None:
        """Register a time duration on the specified tag by providing the start- and end datetime."""
        if not isinstance(tag, str):
            raise TypeError(f"the 'tag' specified was of wrong type {type(tag)}, expected {str}.")
        if not isinstance(start, datetime):
            raise TypeError(f"the 'start' specified was of wrong type {type(start)}, expected {datetime}.")
        if not isinstance(end, datetime):
            raise TypeError(f"the 'end' specified was of wrong type {type(end)}, expected {datetime}.")
        if start > end:
            raise ValueError("the start time was higher, or later, than the end time.")
        duration = end - start
        self.time[tag] = duration.total_seconds()

    def performance_details(self) -> str:
        strings = list()
        for loss_group, loss_values in self.loss.items():
            for loss_name, loss_value in loss_values.items():
                strings.append(f"{loss_group}_{loss_name} {loss_value:.4f}")
        return ", ".join(strings)


class GenerationFullException(Exception):
    pass


class MemberAlreadyExistsError(Exception):
    pass


class Generation(object):
    def __init__(self, dict_constructor=dict, members: Iterator[Checkpoint] = None):
        super().__init__()
        self._members = dict_constructor()
        if members is None:
            return
        for member in members:
            if not isinstance(member, Checkpoint):
                raise TypeError("Members must be of type Checkpoint!")
            if member.uid in self._members:
                raise MemberAlreadyExistsError("Members must be unique!")
            self._members[member.uid] = member

    def __len__(self):
        return len(self._members)

    def __iter__(self):
        return iter(self._members.values())

    def __contains__(self, other):
        return other.uid in self._members

    def __getitem__(self, uid):
        return self._members[uid]

    def __setitem__(self, uid, member):
        self._members[uid] = member

    def entries(self):
        return iter(self._members.items())

    def append(self, member: Checkpoint):
        if member.uid in self._members:
            raise MemberAlreadyExistsError("Member uid already exists.")
        if any(member is gen_member for gen_member in self._members.values()):
            raise MemberAlreadyExistsError("Member object already exists!")
        if not isinstance(member, Checkpoint):
            raise TypeError()
        self._members[member.uid] = member

    def extend(self, iterable):
        if not isinstance(iterable, Iterable):
            raise TypeError()
        [self.append(member) for member in iterable]

    def update(self, member):
        if member.uid not in self._members:
            raise IndexError
        self._members[member.uid] = member

    def remove(self, member):
        if member.uid not in self._members:
            raise IndexError
        del self._members[member.uid]

    def clear(self):
        self._members = dict()
