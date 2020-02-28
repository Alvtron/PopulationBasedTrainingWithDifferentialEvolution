import copy
import torch
import math
import warnings
import shutil

from pathlib import Path
from abc import abstractmethod 
from collections import defaultdict, deque
from itertools import islice, chain
from typing import List, Iterable, Sequence, Generator

from .utils.conversion import dict_to_binary, binary_to_dict
from .utils.iterable import chunks, insert_sequence
from .hyperparameters import ContiniousHyperparameter, _Hyperparameter, Hyperparameters

# extend Path to include copy method
def _copy(self, target):
    assert self.is_file()
    shutil.copy(self, target)
Path.copy = _copy

# global member state directory
MEMBER_STATE_DIRECTORY = Path('.member_states')
# ensure that the state directory is created
MEMBER_STATE_DIRECTORY.mkdir(parents=True, exist_ok=True)

def clear_member_states():
    # delete folder
    shutil.rmtree(MEMBER_STATE_DIRECTORY)
    # create folder
    MEMBER_STATE_DIRECTORY.mkdir(parents=True, exist_ok=True)

def prepare_score(value):
    if hasattr(value, "score"):
        return value.score() if not math.isnan(value.score()) and not math.isinf(value.score()) else None
    elif isinstance(value, (float, int)):
        return value if not math.isnan(value) and not math.isinf(value) else None
    else:
        raise NotImplementedError

class MissingStateError(Exception):
    pass

class MemberState(object):
    '''Base class for member states.'''
    def __init__(self, id, hyper_parameters : Hyperparameters, minimize : bool):
        self.id = id
        self.hyper_parameters = hyper_parameters
        self.minimize = minimize

    @abstractmethod
    def score(self):
        raise NotImplementedError

    def __iter__(self):
        return self.hyper_parameters.parameters()

    def __len__(self) -> int:
        return len(self.hyper_parameters)

    def __getitem__(self, index) -> _Hyperparameter:
        return self.hyper_parameters[index]

    def __setitem__(self, index, value : _Hyperparameter):
        if not isinstance(value, _Hyperparameter):
            raise TypeError()
        self.hyper_parameters[index] = value

    def __str__(self) -> str:
        return f"Member {self.id:03d}"

    def __eq__(self, other) -> bool:
        if other is None or not hasattr(other, id):
            return False
        return self.id == other.id

    def __ne__(self, other) -> bool:
        if other is None or not hasattr(other, id):
            return True
        return self.id != other.id

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

    def update(self, other):
        self.hyper_parameters = copy.deepcopy(other.hyper_parameters)

    def copy(self):
        return copy.deepcopy(self)

class Checkpoint(MemberState):
    '''Class for keeping track of a checkpoint, where the model- and optimizer state are stored in system memory.'''
    def __init__(self, id, hyper_parameters : Hyperparameters, loss_metric : str, eval_metric : str, minimize : bool):
        super().__init__(id, hyper_parameters, minimize)
        self.epochs : int = 0
        self.steps : int = 0
        self.model_state : dict = None
        self.optimizer_state : dict = None
        self.loss_metric : str = loss_metric
        self.eval_metric : str = eval_metric
        self.loss : dict = dict()
        self.time : dict = dict()

    def __str__(self) -> str:
        return super().__str__() + f", epoch {self.epochs:03d}, step {self.steps:05d}"
    
    def __eq__(self, other) -> bool:
        if other is None:
            return False
        return self.id == other.id and self.steps == other.steps

    def __ne__(self, other) -> bool:
        if other is None:
            return True
        return self.id != other.id or self.steps != other.steps

    def score(self) -> float:
        return self.loss['eval'][self.eval_metric] if 'eval' in self.loss and self.eval_metric in self.loss['eval'] else None

    def train_score(self) -> float:
        return self.loss['train'][self.eval_metric] if 'train' in self.loss and self.eval_metric in self.loss['train'] else None

    def test_score(self) -> float:
        return self.loss['test'][self.eval_metric] if 'test' in self.loss and self.eval_metric in self.loss['test'] else None
    
    def __state_directory_path(self) -> Path:
        return Path(MEMBER_STATE_DIRECTORY, f"{self.steps}/{self.id}")

    def __model_state_path(self) -> Path:
        return Path(self.__state_directory_path(), "model_state.pth")

    def __optimizer_state_path(self) -> Path:
        return Path(self.__state_directory_path(), "optimizer_state.pth")

    def has_model_state(self) -> bool:
        """Returns true if the checkpoint has a model state file."""
        return self.__model_state_path().is_file()

    def has_optimizer_state(self) -> bool:
        """Returns true if the checkpoint has a optimizer state file."""
        return self.__optimizer_state_path().is_file()

    def has_state(self) -> bool:
        """Returns true if the checkpoint has any state files."""
        return self.has_model_state() or self.has_optimizer_state() 

    def load_state(self, device='cpu', missing_ok=False):
        """Load the state on the specified device from local files stored in the checkpoint directory. Raises error if the files are not available."""
        if not self.has_model_state():
            if missing_ok:
                self.model_state = None
            else:
                raise MissingStateError(f"Model state file is missing at {self.__model_state_path()}")
        else:
            self.model_state = torch.load(self.__model_state_path(), map_location=device)
        if not self.has_optimizer_state():
            if missing_ok:
                self.optimizer_state = None
            else:
                raise MissingStateError(f"Optimizer state file is missing at {self.__model_state_path()}")
        else:
            self.optimizer_state = torch.load(self.__optimizer_state_path(), map_location=device)

    def unload_state(self):
        """Saves the state locally to file and deletes the in-memory state. Call load() to bring it back into memory."""
        if self.model_state is None:
            raise AttributeError("Can't unload when model state is None. Nothing to unload.")
        if self.optimizer_state is None:
            raise AttributeError("Can't unload when optimizer state is None. Nothing to unload.")
        # ensure state directory created
        self.__state_directory_path().mkdir(parents=True, exist_ok=True)
        # save state objects to file
        torch.save(self.model_state, self.__model_state_path())
        torch.save(self.optimizer_state, self.__optimizer_state_path())
        # delete state objects
        del self.model_state
        del self.optimizer_state
        # clear GPU memory
        torch.cuda.empty_cache()

    def delete_state(self):
        """Deletes the state, both in-memory and any existing local files."""
        # delete state in memory
        self.model_state = None
        self.optimizer_state = None
        # delete state files if they exist
        if self.has_model_state():
            self.__model_state_path().unlink()
        if self.has_optimizer_state():
            self.__optimizer_state_path().unlink()

    def copy_state(self, other):
        """Copy the state of the other checkpoint."""
        if hasattr(other, 'model_state') and other.model_state is not None:
            self.model_state = copy.deepcopy(other.model_state)
        elif other.has_model_state():
            other.__model_state_path().copy(self.__model_state_path())
        if hasattr(other, 'optimizer_state') and other.optimizer_state is not None:
            self.optimizer_state = copy.deepcopy(other.optimizer_state)
        elif other.has_optimizer_state():
            other.__optimizer_state_path().copy(self.__optimizer_state_path())

    def update(self, other):
        """
        Replace own hyper-parameters, model state and optimizer state from the provided checkpoint.
        The step and epoch counts are also copied. Resets loss and time.
        """
        self.epochs = other.epochs
        self.steps = other.steps
        self.hyper_parameters = copy.deepcopy(other.hyper_parameters)
        if self != other:
            self.copy_state(other)
        self.loss = dict()
        self.time = dict()

    def performance_details(self) -> str:
        strings = list()
        for loss_group, loss_values in self.loss.items():
            for loss_name, loss_value in loss_values.items():
                strings.append(f"{loss_group}_{loss_name} {loss_value:.5f}")
        return ", ".join(strings)

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

    def extend(self, generation : Generation):
        if (self.size != generation.size):
            raise ValueError("generation sizes do not match.")
        self.extend(generation)

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