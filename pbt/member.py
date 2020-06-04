import gc
import copy
import math
import shutil
import warnings
from pathlib import Path
from abc import abstractmethod
from collections import defaultdict, deque
from itertools import islice, chain
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
    elif hasattr(value, "eval_score"):
        return prepare_score(value.eval_score())
    else:
        raise NotImplementedError


class MissingStateError(Exception):
    pass


class Checkpoint(object):
    '''Class for keeping track of a checkpoint, where the model- and optimizer state are stored in system memory.'''

    def __init__(self, id, parameters: Hyperparameters, loss_metric: str, eval_metric: str, minimize: bool):
        self.TRAIN_LABEL = 'train'
        self.EVAL_LABEL = 'eval'
        self.TEST_LABEL = 'test'
        self.MODEL_STATE_PROPERTY = 'model_state'
        self.OPTIMIZER_STATE_PROPERTY = 'optimizer_state'
        self.id: object = id
        self.parameters: Hyperparameters = parameters
        self.minimize: bool = minimize
        self.steps: int = 0
        self.epochs: int = 0
        self.model_state: dict = None
        self.optimizer_state: dict = None
        self.loss_metric: str = loss_metric
        self.eval_metric: str = eval_metric
        self.loss: Dict[str, dict] = dict()
        self.time: Dict[str, dict] = dict()

    def __str__(self) -> str:
        return f"member {self.id}, step {self.steps}, epoch {self.epochs}"

    def __getitem__(self, index: int) -> float:
        return self.parameters[index].normalized

    def __setitem__(self, index: int, value: float):
        if not isinstance(value, float):
            raise TypeError()
        self.parameters[index].normalized = value

    def __eq__(self, other) -> bool:
        if other is None or type(self) != type(other):
            return False
        return self.id == other.id and self.steps == other.steps

    def __ne__(self, other) -> bool:
        if other is None or type(self) != type(other):
            return True
        return self.id != other.id or self.steps != other.steps

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
        if self.TRAIN_LABEL not in self.loss or self.eval_metric not in self.loss[self.TRAIN_LABEL]:
            return None
        return self.loss[self.TRAIN_LABEL][self.eval_metric]

    def eval_score(self) -> float:
        if self.EVAL_LABEL not in self.loss or self.eval_metric not in self.loss[self.EVAL_LABEL]:
            return None
        return self.loss[self.EVAL_LABEL][self.eval_metric]

    def test_score(self) -> float:
        if self.TEST_LABEL not in self.loss or self.eval_metric not in self.loss[self.TEST_LABEL]:
            return None
        return self.loss[self.TEST_LABEL][self.eval_metric]

    def copy_score(self, other) -> None:
        """Replace own score with score from the other member."""
        self.loss = copy.deepcopy(other.loss)

    def has_state(self) -> bool:
        return hasattr(self, self.MODEL_STATE_PROPERTY) and self.model_state is not None and hasattr(self, self.OPTIMIZER_STATE_PROPERTY) and self.optimizer_state is not None

    def load_state(self, device: str = 'cpu', missing_ok: bool = False):
        """Move all tensors in state to specified device. Call unload_state to move state back to cpu. Raises error if states are not available."""
        # load model state
        if self.model_state is not None:
            modify_iterable(self.model_state, lambda x: x.to(
                device), lambda x: isinstance(x, torch.Tensor))
        elif missing_ok:
            self.model_state = None
        else:
            raise MissingStateError(f"Model state is missing.")
        # load optimizer state
        if self.optimizer_state is not None:
            modify_iterable(self.optimizer_state, lambda x: x.to(
                device), lambda x: isinstance(x, torch.Tensor))
        elif missing_ok:
            self.optimizer_state = None
        else:
            raise MissingStateError(f"Optimizer state is missing.")

    def unload_state(self):
        """Move all tensors in state to cpu. Call load_state to load state on a specific device. Raises error if states are not available."""
        device = 'cpu'
        if self.model_state is None or not hasattr(self, self.MODEL_STATE_PROPERTY):
            raise AttributeError(
                "Can't unload when model state is None. Nothing to unload.")
        if self.optimizer_state is None or not hasattr(self, self.OPTIMIZER_STATE_PROPERTY):
            raise AttributeError(
                "Can't unload when optimizer state is None. Nothing to unload.")
        modify_iterable(self.model_state, lambda x: x.to(
            device), lambda x: isinstance(x, torch.Tensor))
        modify_iterable(self.optimizer_state, lambda x: x.to(
            device), lambda x: isinstance(x, torch.Tensor))

    def delete_state(self):
        """Deletes the states from memory."""
        # delete state in memory
        if hasattr(self, self.MODEL_STATE_PROPERTY):
            del self.model_state
        if hasattr(self, self.OPTIMIZER_STATE_PROPERTY):
            del self.optimizer_state

    def copy_state(self, other):
        """Replace own model state and optimizer state with the ones from the other checkpoint."""
        if self == other:
            return
        # copy model state
        if hasattr(other, self.MODEL_STATE_PROPERTY):
            assert other.model_state is not None, "Attempted to copy empty model state"
            self.model_state = copy.deepcopy(other.model_state)
        # copy optimizer state
        if hasattr(other, self.OPTIMIZER_STATE_PROPERTY):
            assert other.optimizer_state is not None, "Attempted to copy empty optimizer state"
            self.optimizer_state = copy.deepcopy(other.optimizer_state)

    def copy_parameters(self, other):
        """Replace own hyper-parameters with the ones from the other checkpoint."""
        self.parameters = copy.deepcopy(other.parameters)

    def copy(self):
        return copy.deepcopy(self)

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
            if member.id in self._members:
                raise MemberAlreadyExistsError("Members must be unique!")
            self._members[member.id] = member

    def __len__(self):
        return len(self._members)

    def __iter__(self):
        return iter(self._members.values())

    def __contains__(self, other):
        return other.id in self._members

    def __getitem__(self, id):
        return self._members[id]

    def __setitem__(self, id, member):
        self._members[id] = member

    def entries(self):
        return iter(self._members.items())

    def append(self, member: Checkpoint):
        if member.id in self._members:
            raise MemberAlreadyExistsError("Member id already exists.")
        if any(member is gen_member for gen_member in self._members.values()):
            raise MemberAlreadyExistsError("Member object already exists!")
        if not isinstance(member, Checkpoint):
            raise TypeError()
        self._members[member.id] = member

    def extend(self, iterable):
        if not isinstance(iterable, Iterable):
            raise TypeError()
        [self.append(member) for member in iterable]

    def update(self, member):
        if member.id not in self._members:
            raise IndexError
        self._members[member.id] = member

    def remove(self, member):
        if member.id not in self._members:
            raise IndexError
        del self._members[member.id]

    def clear(self):
        self._members = dict()
