import copy
import torch
import math
import warnings

from pathlib import Path
from abc import abstractmethod 
from collections import defaultdict, deque
from itertools import islice, chain
from typing import List, Iterable, Sequence, Generator

from .utils.conversion import dict_to_binary, binary_to_dict
from .utils.iterable import chunks, insert_sequence
from .hyperparameters import Hyperparameter, Hyperparameters

class MemberState(object):
    '''Base class for member states.'''
    def __init__(self, id, hyper_parameters : Hyperparameters, minimize : bool):
        self.id = id
        self.hyper_parameters : Hyperparameters = hyper_parameters
        self.minimize : bool = minimize

    @abstractmethod
    def score(self):
        pass

    def __iter__(self):
        return self.hyper_parameters.parameters()

    def __len__(self):
        return len(self.hyper_parameters)

    def __getitem__(self, index):
        return self.hyper_parameters[index]

    def __setitem__(self, index, value : Hyperparameter):
        if not isinstance(value, Hyperparameter):
            raise TypeError()
        self.hyper_parameters[index] = value

    def __str__(self):
        return f"Member {self.id:03d}"

    def __eq__(self, other):
        if other is None:
            return False
        return self.id == other.id

    def __ne__(self, other):
        if other is None:
            return True
        return self.id != other.id

    def __lt__(self, other):
        if other is None:
            return False
        if isinstance(other, (float, int)):
            return self.score() > other if self.minimize else self.score() < other
        elif isinstance(other, MemberState):
            return self.score() > other.score() if self.minimize else self.score() < other.score()
        else:
            raise NotImplementedError()

    def __le__(self, other):
        if other is None:
            return False
        if isinstance(other, (float, int)):
            return self.score() >= other if self.minimize else self.score() <= other
        elif isinstance(other, MemberState):
            return self.score() >= other.score() if self.minimize else self.score() <= other.score()
        else:
            raise NotImplementedError()

    def __gt__(self, other):
        if other is None:
            return True
        if isinstance(other, (float, int)):
            return self.score() < other if self.minimize else self.score() > other
        elif isinstance(other, MemberState):
            return self.score() < other.score() if self.minimize else self.score() > other.score()
        else:
            raise NotImplementedError()

    def __ge__(self, other):
        if other is None:
            return True
        if isinstance(other, (float, int)):
            return self.score() <= other if self.minimize else self.score() >= other
        elif isinstance(other, MemberState):
            return self.score() <= other.score() if self.minimize else self.score() >= other.score()
        else:
            raise NotImplementedError()

    def update(self, other):
        self.hyper_parameters = copy.deepcopy(other.hyper_parameters)

    def copy(self):
        return copy.deepcopy(self)

class Checkpoint(MemberState):
    '''Class for keeping track of a checkpoint, where the model- and optimizer state are stored locally on a file instead of in memory.'''
    def __init__(self, id, directory: Path, hyper_parameters : Hyperparameters, loss_metric : str, eval_metric : str, minimize : bool, step_size : int = 1):
        super().__init__(id, hyper_parameters, minimize)
        self.epochs : int = 0
        self.steps : int = 0
        self.model_state_load_path : Path = None
        self.optimizer_state_load_path : Path = None
        self.directory : Path = directory
        self.step_size : int = step_size
        self.loss_metric : str = loss_metric
        self.eval_metric : str = eval_metric
        self.loss : dict = dict()
        self.time : dict = dict()
        # ensure directory is created
        self.directory.mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        return super().__str__() + f", epoch {self.epochs:03d}, step {self.steps:05d}"
    
    def __eq__(self, other) -> bool:
        if other is None:
            return False
        return self.id == other.id and self.steps == other.steps and self.step_size == other.step_size

    def __ne__(self, other) -> bool:
        if other is None:
            return True
        return self.id != other.id or self.steps != other.steps or self.step_size != other.step_size

    def score(self) -> float:
        return self.loss['eval'][self.eval_metric]

    @property
    def model_state_save_path(self) -> Path:
        return Path(self.directory, f"{self.steps:05d}_model_state.pth")

    @property 
    def optimizer_state_save_path(self) -> Path:
        return Path(self.directory, f"{self.steps:05d}_optimizer_state.pth")

    def train_score(self) -> float:
        return self.loss['train'][self.eval_metric]

    def eval_score(self) -> float:
        return self.score()

    def test_score(self) -> float:
        return self.loss['test'][self.eval_metric]

    def has_model_state(self) -> bool:
        return self.model_state_load_path and self.model_state_load_path.is_file()

    def has_optimizer_state(self) -> bool:
        return self.optimizer_state_load_path and self.optimizer_state_load_path.is_file()

    def has_state(self) -> bool:
        return self.has_model_state() and self.has_optimizer_state() 
    
    def performance_details(self) -> str:
        strings = list()
        for loss_group, loss_values in self.loss.items():
            for loss_name, loss_value in loss_values.items():
                strings.append(f"{loss_group}_{loss_name} {loss_value:.5f}")
        return ", ".join(strings)

    def load_state(self):
        model_state = torch.load(self.model_state_load_path) if self.has_model_state() else None
        optimizer_state = torch.load(self.optimizer_state_load_path) if self.has_optimizer_state() else None
        return model_state, optimizer_state

    def save_state(self, model_state, optimizer_state):
        torch.save(model_state, self.model_state_save_path)
        torch.save(optimizer_state, self.optimizer_state_save_path)
        self.model_state_load_path = self.model_state_save_path
        self.optimizer_state_load_path = self.optimizer_state_save_path

    def delete_state(self):
        if self.has_model_state() and self.model_state_load_path.parent == self.directory:
            self.model_state_load_path.unlink()
        if self.has_optimizer_state() and self.optimizer_state_load_path.parent == self.directory:
            self.optimizer_state_load_path.unlink()

    def update(self, other):
        """
        Replace own hyper-parameters, model state and optimizer state from the provided checkpoint.\n
        Resets loss and time.
        """
        self.hyper_parameters = copy.deepcopy(other.hyper_parameters)
        self.model_state_load_path = copy.deepcopy(other.model_state_load_path)
        self.optimizer_state_load_path = copy.deepcopy(other.optimizer_state_load_path)
        self.loss = dict()
        self.time = dict()

class GenerationFullException(Exception):
    pass

class Generation(list):
    def __init__(self, *args, size):
        list.__init__(self, *args)
        self.size = size

    def full(self) -> bool:
        return self.__len__() >= self.size

    def insert(self, index, value):
        if not isinstance(value, MemberState):
            raise TypeError()
        if self.full():
            raise GenerationFullException()
        super().insert(index, value)
    
class Population(Generation):
    def __init__(self, size):
        super().__init__(self, size=size)
        self.__history : List[Generation] = list()

    @property
    def generations(self) -> List[Generation]:
        """ Return all generations in the population."""
        return self.__history + [self.__to_generation()]

    @property
    def members(self) -> List[MemberState]:
        """ Return all members across all generations in the population."""
        return list(chain.from_iterable(self.generations))

    def new_generation(self):
        """Create new generation. This saves the old generation"""
        if not self.full():
            warnings.warn("Current population is not full.")
        self.__history.append(self.__to_generation())
        self.clear()

    def __to_generation(self) -> Generation:
        return Generation(self, size=self.size)

class PopulationAsync(Generation):
    def __init__(self, size):
        super().__init__(self, size=size)
        self.__history : List[MemberState] = list()
        self.__ban = list()

    @property
    def generations(self) -> List[Generation]:
        """ Return all generations in the population."""
        return self.__history + [self.__to_generation()]

    @property
    def members(self) -> List[MemberState]:
        """ Return all members across all generations in the population."""
        return self.__history + self

    def append(self, member : MemberState):
        if self.full():
            self.__history.append(self.pop())
        self.append(member)

    def __to_generation(self) -> Generation:
        return Generation(self, size=self.size)