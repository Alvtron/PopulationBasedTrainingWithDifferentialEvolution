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
    def __init__(self, id, parameters : Hyperparameters, minimize : bool):
        self.id = id
        self.parameters = parameters
        self.minimize = minimize

    @abstractmethod
    def score(self):
        raise NotImplementedError

    def __getitem__(self, index : int) -> float:
        return self.parameters[index].normalized

    def __setitem__(self, index : int, value : float):
        if not isinstance(value, float):
            raise TypeError()
        self.parameters[index].normalized = value

    def __str__(self) -> str:
        return f"Member {self.id:03d}"

    def __eq__(self, other) -> bool:
        if other is None or not hasattr(other, 'id'):
            return False
        return self.id == other.id

    def __ne__(self, other) -> bool:
        if other is None or not hasattr(other, 'id'):
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

    def copy_parameters(self, other):
        """Replace own hyper-parameters with the ones from the other checkpoint."""
        self.parameters = copy.deepcopy(other.parameters)

    def copy(self):
        return copy.deepcopy(self)

class Checkpoint(MemberState):
    '''Class for keeping track of a checkpoint, where the model- and optimizer state are stored in system memory.'''
    def __init__(self, id, parameters : Hyperparameters, loss_metric : str, eval_metric : str, minimize : bool):
        super().__init__(id, parameters, minimize)
        self.epochs : int = 0
        self.steps : int = 0
        self.model_state : dict = None
        self.optimizer_state : dict = None
        self.loss_metric : str = loss_metric
        self.eval_metric : str = eval_metric
        self.loss : Dict[object, dict] = dict()
        self.time : Dict[object, dict] = dict()

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

    def has_model_state_files(self) -> bool:
        """Returns true if the checkpoint has a model state file."""
        return self.__model_state_path().is_file()

    def has_optimizer_state_files(self) -> bool:
        """Returns true if the checkpoint has a optimizer state file."""
        return self.__optimizer_state_path().is_file()

    def has_state_files(self) -> bool:
        """Returns true if the checkpoint has any state files."""
        return self.has_model_state_files() or self.has_optimizer_state_files() 

    def load_state(self, device : str = 'cpu', missing_ok : bool = False):
        """Load the state on the specified device from local files stored in the checkpoint directory. Raises error if the files are not available."""
        # load model state
        if self.has_model_state_files():
            self.model_state = torch.load(self.__model_state_path(), map_location=device)
        else:
            if missing_ok:
                self.model_state = None
            else:
                raise MissingStateError(f"Model state file is missing at {self.__model_state_path()}")
        # load optimizer state
        if self.has_optimizer_state_files():
            self.optimizer_state = torch.load(self.__optimizer_state_path(), map_location=device)
        else:
            if missing_ok:
                self.optimizer_state = None
            else:
                raise MissingStateError(f"Optimizer state file is missing at {self.__model_state_path()}")

    def unload_state(self, device : str = 'cpu'):
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

    def delete_state(self):
        """Deletes the state, both in-memory and any existing local files."""
        # delete state in memory
        if hasattr(self, 'model_state'):
            del self.model_state
        if hasattr(self, 'optimizer_state'):
            del self.optimizer_state
        # delete state files if they exist
        if self.has_model_state_files():
            self.__model_state_path().unlink()
        if self.has_optimizer_state_files():
            self.__optimizer_state_path().unlink()

    def copy_state(self, other):
        """Replace own model state and optimizer state with the ones from the other checkpoint."""
        if self == other:
            return
        # copy model state
        if hasattr(other, 'model_state') and other.model_state is not None:
            self.model_state = copy.deepcopy(other.model_state)
        elif other.has_model_state_files():
            other.__model_state_path().copy(self.__model_state_path())
        # copy optimizer state
        if hasattr(other, 'optimizer_state') and other.optimizer_state is not None:
            self.optimizer_state = copy.deepcopy(other.optimizer_state)
        elif other.has_optimizer_state_files():
            other.__optimizer_state_path().copy(self.__optimizer_state_path())

    def move_state(self, device : str):
        if hasattr(self, 'model_state') and self.model_state is not None:
            modify_iterable(self.model_state, lambda x: x.to(device) , lambda x: isinstance(x, torch.Tensor))
        if hasattr(self, 'optimizer_state') and self.optimizer_state is not None:
            modify_iterable(self.optimizer_state, lambda x: x.to(device) , lambda x: isinstance(x, torch.Tensor))

    def performance_details(self) -> str:
        strings = list()
        for loss_group, loss_values in self.loss.items():
            for loss_name, loss_value in loss_values.items():
                strings.append(f"{loss_group}_{loss_name} {loss_value:.5f}")
        return ", ".join(strings)

class GenerationFullException(Exception):
    pass

class MemberAlreadyExistsError(Exception):
    pass

class Generation(object):
    def __init__(self, members : Iterator[MemberState] = None):
        super().__init__()
        self._members : Dict[MemberState] = dict()
        if members is None:
            return
        for member in members:
            if not isinstance(member, MemberState):
                raise TypeError("Members must be of type MemberState!")
            if member.id in self._members:
                raise MemberAlreadyExistsError("Members must be unique!")
            self._members[member.id] = member

    def __len__(self):
        return len(self._members)

    def __iter__(self):
        return iter(self._members.values())

    def __getitem__(self, id):
        return self._members[id]

    def __setitem__(self, id, member):
        self._members[id] = member

    def append(self, member : MemberState):
        if member.id in self._members:
            raise MemberAlreadyExistsError("Member id already exists.")
        if any(member is gen_member for gen_member in self._members.values()):
            raise MemberAlreadyExistsError("Member object already exists!")
        if not isinstance(member, MemberState):
            raise TypeError()
        self._members[member.id] = member

    def extend(self, iterable):
        if not isinstance(iterable, Iterable):
            raise TypeError()
        [self.append(member) for member in iterable]

    def update(self, member):
        if member.id not in self._members:
            raise IndexError
        if member != self._members[member.id]:
            raise ValueError(f"Member does not match existing member at id {member.id}.")
        self._members[member.id] = member
    
    def remove(self, member):
        if member.id not in self._members:
            raise IndexError
        del self._members[member.id]

    def clear(self):
        self._members = dict()

class Population(object):
    def __init__(self):
        super().__init__()
        self.generations : List[Generation] = list()

    @property
    def current(self) -> Generation:
        return self.generations[-1]

    @property
    def members(self) -> List[MemberState]:
        """ Return all members across all generations in the population."""
        return list(chain.from_iterable(self.generations))

    def append(self, generation : Generation):
        if not isinstance(generation, Generation):
            raise TypeError
        if any(generation is existing for existing in self.generations):
            raise ValueError("Attempting to add the same generation twice!")
        self.generations.append(generation)