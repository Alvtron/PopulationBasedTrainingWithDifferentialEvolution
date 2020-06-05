from __future__ import annotations
import random
import math
import copy
import warnings
from functools import partial
from typing import Dict, Union, Tuple, Iterable, TypeVar, Generic
from abc import abstractmethod

from .utils.constraint import translate, clip, reflect
from .utils.iterable import value_by_fraction

HP_TYPE = TypeVar('ParameterType')

class InvalidSearchSpaceException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class _Hyperparameter(object):
    '''
    Class for creating and storing a hyperparameter in a given, constrained search space.
    '''
    def __init__(self, *args: Iterable[HP_TYPE], value: HP_TYPE = None, constraint: str = 'clip') -> None:
        ''' 
        Provide a set of [lower bound, upper bound] as float/int, or categorical elements [obj1, obj2, ..., objn].
        Sets the search space and samples a new candidate from an uniform distribution.
        '''
        if args == None:
            raise ValueError("No arguments provided.")
        if not isinstance(constraint, str):
            raise TypeError(f"the 'constraint' specified was of wrong type {type(constraint)}, expected {str}.")
        self.MIN_NORM: float = 0.0
        self.MAX_NORM: float = 1.0
        self.set_constraint(constraint)
        self.search_space: Tuple[HP_TYPE, ...] = tuple(args)
        self._normalized = self.from_value(value) if value is not None else random.uniform(self.MIN_NORM, self.MAX_NORM)

    def __repr__(self):
        return repr(self.value)

    def _translate_from_norm(self, normalized_value: float) -> HP_TYPE:
        return translate(normalized_value, self.MIN_NORM, self.MAX_NORM, self.lower_bound, self.upper_bound)
    
    def _translate_from_value(self, value: HP_TYPE) -> float:
        return translate(value, self.lower_bound, self.upper_bound, self.MIN_NORM, self.MAX_NORM)

    def set_constraint(self, constraint: str) -> None:
        if isinstance(constraint, str):
            if constraint == 'clip':
                self._constrain = partial(clip, min_value=self.MIN_NORM, max_value=self.MAX_NORM)
            elif constraint == 'reflect':
                self._constrain = partial(reflect, min_value=self.MIN_NORM, max_value=self.MAX_NORM)
            else:
                raise NotImplementedError(f"No constraint matches '{constraint}'")
        elif callable(constraint):
            self._constrain = partial(constraint, min_value=self.MIN_NORM, max_value=self.MAX_NORM)
        else:
            raise ValueError("The provided constraint must be of type str or callable.")
        

    def __str__(self) -> str:
        return f"v: {self.value:.4E}, n: {self.normalized:.4f} U({self.lower_bound},{self.upper_bound})"

    @property
    def normalized(self) -> float:
        """Returns the normalized hyperparameter value."""
        return self._normalized

    @normalized.setter
    def normalized(self, value: float) -> float:
        """Sets the normalized hyperparameter value."""
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError("value must be a finite value of type float.")
        self._normalized = self._constrain(value)
    
    @property
    @abstractmethod
    def value(self) -> HP_TYPE:
        raise NotImplementedError

    @value.setter
    @abstractmethod
    def value(self, value: HP_TYPE):
        raise NotImplementedError

    @property
    @abstractmethod
    def lower_bound(self) -> Union[int, float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def upper_bound(self) -> Union[int, float]:
        raise NotImplementedError

    @abstractmethod
    def from_value(self, value: HP_TYPE) -> float:
        raise NotImplementedError

    @abstractmethod
    def from_normalized(self, normalized_value: float) -> HP_TYPE:
        raise NotImplementedError

    def sample_uniform(self) -> None:
        ''' Samples a new candidate from an uniform distribution bound by the lower and upper bounds. '''
        self._normalized = random.uniform(self.MIN_NORM, self.MAX_NORM)

    def equal_search_space(self, other) -> bool:
        """Return true if the search space is equal."""
        return isinstance(other, self.__class__) and self.search_space == other.search_space

    def __add__(self, other: Union[int, float, HP_TYPE]):
        new_hp = copy.deepcopy(self)
        if isinstance(other, self.__class__):
            if not new_hp.equal_search_space(other):
                raise ValueError("Addition is not supported for hyperparameters of unequal search spaces.")
            new_hp._normalized = self._constrain(new_hp._normalized + other._normalized)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp._normalized = self._constrain(new_hp._normalized + other)
            return new_hp
        else:
            raise TypeError(f"Addition is only supported for values of type {self.__class__}, {float} or {int}.")

    def __sub__(self, other: Union[int, float, HP_TYPE]):
        new_hp = copy.deepcopy(self)
        if isinstance(other, self.__class__):
            if not new_hp.equal_search_space(other):
                raise ValueError("Subtraction is not supported for hyperparameters of unequal search spaces.")
            new_hp._normalized = self._constrain(new_hp._normalized - other._normalized)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp._normalized = self._constrain(new_hp._normalized - other)
            return new_hp
        else:
            raise TypeError(f"Subtraction is only supported for values of type {self.__class__}, {float} or {int}.")

    def __mul__(self, other: Union[int, float, HP_TYPE]):
        new_hp = copy.deepcopy(self)
        if isinstance(other, self.__class__):
            if not new_hp.equal_search_space(other):
                raise ValueError("Multiplication is not supported for hyperparameters of unequal search spaces.")
            new_hp._normalized = self._constrain(new_hp._normalized * other._normalized)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp._normalized = self._constrain(new_hp._normalized * other)
            return new_hp
        else:
            raise TypeError(f"Multiplication is only supported for values of type {self.__class__}, {float} or {int}.")

    def __truediv__(self, other: Union[int, float, HP_TYPE]):
        new_hp = copy.deepcopy(self)
        if isinstance(other, self.__class__):
            if not new_hp.equal_search_space(other):
                raise ValueError("Divition is not supported for hyperparameters of unequal search spaces.")
            new_hp._normalized = self._constrain(new_hp._normalized / other._normalized)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp._normalized = self._constrain(new_hp._normalized / other)
            return new_hp
        else:
            raise TypeError(f"Divition is only supported for values of type {self.__class__}, {float} or {int}.")

    def __iadd__(self, other: Union[int, float, HP_TYPE]):
        if isinstance(other, self.__class__):
            if not self.equal_search_space(other):
                raise ValueError("Addition is not supported for hyperparameters of unequal search spaces.")
            self._normalized = self._constrain(self._normalized + other._normalized)
        elif isinstance(other, (float, int)):
            self._normalized = self._constrain(self._normalized + other)
        else:
            raise TypeError(f"Addition is only supported for values of type {self.__class__}, {float} or {int}.")
        return self

    def __isub__(self, other: Union[int, float, HP_TYPE]):
        if isinstance(other, self.__class__):
            if not self.equal_search_space(other):
                raise ValueError("Subtraction is not supported for hyperparameters of unequal search spaces.")
            self._normalized = self._constrain(self._normalized - other._normalized)
        elif isinstance(other, (float, int)):
            self._normalized = self._constrain(self._normalized - other)
        else:
            raise TypeError(f"Subtraction is only supported for values of type {self.__class__}, {float} or {int}.")
        return self

    def __imul__(self, other: Union[int, float, HP_TYPE]):
        if isinstance(other, self.__class__):
            if not self.equal_search_space(other):
                raise ValueError("Multiplication is not supported for hyperparameters of unequal search spaces.")
            self._normalized = self._constrain(self._normalized * other._normalized)
        elif isinstance(other, (float, int)):
            self._normalized = self._constrain(self._normalized * other)
        else:
            raise TypeError(f"Multiplication is only supported for values of type {self.__class__}, {float} or {int}.")
        return self

    def __idiv__(self, other: Union[int, float, HP_TYPE]):
        if isinstance(other, self.__class__):
            if not self.equal_search_space(other):
                raise ValueError("Divition is not supported for hyperparameters of unequal search spaces.")
            self._normalized = self._constrain(self._normalized / other._normalized)
        elif isinstance(other, (float, int)):
            self._normalized = self._constrain(self._normalized / other)
        else:
            raise TypeError(f"Divition is only supported for values of type {self.__class__}, {float} or {int}.")
        return self

    def __lt__(self, other: self.__class__):
        if isinstance(other, self.__class__) and self.search_space == other.search_space:
            return self._normalized < other._normalized
        else:
            raise TypeError(f"Comparison operations is supported for values of type {self.__class__} in equal search space.")

    def __gt__(self, other: self.__class__):
        if isinstance(other, self.__class__) and self.search_space == other.search_space:
            return self._normalized > other._normalized
        else:
            raise TypeError(f"Comparison operations is supported for values of type {self.__class__} in equal search space.")

    def __le__(self, other: self.__class__):
        if isinstance(other, self.__class__) and self.search_space == other.search_space:
            return self._normalized <= other._normalized
        else:
            raise TypeError(f"Comparison operations is supported for values of type {self.__class__} in equal search space.")

    def __ge__(self, other: self.__class__):
        if isinstance(other, self.__class__) and self.search_space == other.search_space:
            return self._normalized >= other._normalized
        else:
            raise TypeError(f"Comparison operations is supported for values of type {self.__class__} in equal search space.")

    def __eq__(self, other: self.__class__):
        if isinstance(other, self.__class__):
            return self.search_space == other.search_space and self._normalized == other._normalized
        else:
            raise TypeError(f"Comparison operations is supported for values of type {self.__class__} in equal search space.")

    def __ne__(self, other: self.__class__):
        if isinstance(other, self.__class__):
            return self.search_space != other.search_space or self._normalized != other._normalized
        else:
            raise TypeError(f"Comparison operations is supported for values of type {self.__class__} in equal search space.")

class ContiniousHyperparameter(_Hyperparameter):
    '''
    Class for creating and storing a hyperparameter in a given, constrained search space.
    '''
    def __init__(self, minimum: Union[int, float], maximum: Union[int, float], value: Union[int, float] = None, constraint: str = 'clip'):
        ''' 
        Provide a set of [lower bound, upper bound] as float/int.
        Sets the search space and samples a new candidate from an uniform distribution.
        '''
        if not isinstance(minimum, (float, int)) or not isinstance(maximum, (float, int)):
            raise TypeError(f"Continious hyperparameters must be of type {float} or {int}.")
        if minimum > maximum:
            raise ValueError("The minimum must be lower than the maximum.")
        if value and not isinstance(value, (float, int)):
            raise TypeError(f"Continious hyperparameters must be of type {float} or {int}.")
        if value and not (minimum <= value <= maximum):
            raise ValueError(f"The provided value must be in range [{minimum},{maximum}].")
        super().__init__(minimum, maximum, value=value, constraint=constraint)

    @property
    def value(self) -> Union[int, float]:
        """Returns the representative hyperparameter value."""
        if self._normalized == None:
            raise ValueError("Developer error. '_normalized' is None.")
        return self.from_normalized(self._normalized)

    @value.setter
    def value(self, value: Union[int, float]):
        """Sets the hyperparameter value."""
        if not(self.lower_bound <= value <= self.upper_bound):
            warnings.warn(f"The value {value} is outside the search space U({self.lower_bound}, {self.upper_bound}). The value will be constrained.")
        self._normalized = self._constrain(self.from_value(value))

    @property
    def lower_bound(self) -> Union[int, float]:
        ''' Returns the lower bounds of the hyper-parameter search space. If categorical, return the first search space index. '''
        return self.search_space[0]

    @property 
    def upper_bound(self) -> Union[int, float]:
        ''' Returns the upper bounds of the hyper-parameter search space. If categorical, return the last search space index. '''
        return self.search_space[-1]

    def from_value(self, value: Union[int, float]) -> float:
        """Returns a normalized version of the provided value."""
        if isinstance(value, (int, float)):
            return self._translate_from_value(value)
        else:
            raise Exception(f"Non-categorical hyperparameters must be of type {float} or {int}.")

    def from_normalized(self, normalized_value: float) -> Union[int, float]:
        """Returns a search space value from the provided normalized value."""
        constrained = self._constrain(normalized_value)
        trainslated = self._translate_from_norm(constrained)
        if isinstance(self.search_space[0], float):
            return float(trainslated)
        elif isinstance(self.search_space[0], int):
            return int(round(trainslated))
        else:
            raise Exception(f"Non-categorical hyperparameters must be of type {float} or {int}.")

class DiscreteHyperparameter(_Hyperparameter):
    def __init__(self, *search_space: Iterable[HP_TYPE], value: HP_TYPE = None, constraint: str = 'clip'):
        ''' 
        Provide a set of categorical elements [obj1, obj2, ..., objn].
        Sets the search space and samples a new candidate from an uniform distribution.
        '''
        if not search_space:
            raise ValueError("No search space provided.")
        if value is not None and value not in search_space:
            raise ValueError("The provided value must be present in the provided categorical search space.")
        super().__init__(*search_space, value=value, constraint=constraint)
    
    @property
    def value(self) -> HP_TYPE:
        """Returns the representative hyperparameter value."""
        if self._normalized == None:
            raise ValueError("Developer error. '_normalized' is None.")
        return self.from_normalized(self._normalized)

    @value.setter
    def value(self, value: HP_TYPE):
        """Sets the hyperparameter value."""
        if value not in self.search_space:
            raise ValueError("The provided value must be present in the categorical search space.")
        self._normalized = self._constrain(self.from_value(value))

    @property
    def lower_bound(self) -> int:
        ''' Returns the lower bounds of the hyper-parameter search space. For categorical, it returns the first search space index. '''
        return 0

    @property 
    def upper_bound(self) -> int:
        ''' Returns the upper bounds of the hyper-parameter search space. For categorical, it returns the last search space index. '''
        return len(self.search_space) - 1

    def from_value(self, value: HP_TYPE) -> float:
        """Returns a normalized version of the provided value."""
        assert value in self.search_space, f"The provided value {value} does not exist within the categorical search space."
        index = self.search_space.index(value)
        return self._translate_from_value(index)

    def from_normalized(self, normalized_value: float) -> HP_TYPE:
        """Returns a search space value from the provided normalized value."""
        constrained = self._constrain(normalized_value)
        return value_by_fraction(self.search_space, constrained)

    def equal_search_space(self, other) -> bool:
        return isinstance(other, self.__class__) and super().equal_search_space(other)

    def __lt__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Comparison operations are only supported for values of type {self.__class__}.")
        return super().__lt__(other)

    def __gt__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Comparison operations are only supported for values of type {self.__class__}.")
        return super().__gt__(other)

    def __le__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Comparison operations are only supported for values of type {self.__class__}.")
        return super().__le__(other)

    def __ge__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Comparison operations are only supported for values of type {self.__class__}.")
        return super().__ge__(other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Comparison operations are only supported for values of type {self.__class__}.")
        return super().__eq__(other)

    def __ne__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Comparison operations are only supported for values of type {self.__class__}.")
        return super().__ne__(other)

class Hyperparameters(object):
    ''' Class for storing and updating hyperparameters. '''
    def __init__(self, **hp_groups: Dict[str, Dict[str, _Hyperparameter]]):
        if not hp_groups or sum(1 for hp_dict in hp_groups.values() if hp_dict is not None) == 0:
            raise TypeError(f"At least one argument required!")
        for group, hp_dict in hp_groups.items():
            if hp_dict is None:
                continue
            if isinstance(hp_dict, dict) and not all(isinstance(hp_name, str) and isinstance(hp_value, _Hyperparameter) for hp_name, hp_value in hp_dict.items()):
                raise TypeError(f"arguments must be one or more dicts of type Dict[str, _Hyperparameter].")
            setattr(self, group, hp_dict)

    def __str__(self) -> str:
        return ''.join(f"{hp_name}: {hp_value}\n" for hp_name, hp_value in self.items())

    def __eq__(self, other) -> bool:
        if not hasattr(other, '__iter__'):
            return False
        elif not hasattr(other, 'items'):
            return False
        elif any(not isinstance(hp, _Hyperparameter) for hp in other):
            return False
        elif any(self_hp != other_hp for self_hp, other_hp in zip(self, other)):
            return False
        elif any(self_key != other_key for self_key, other_key in zip(self.keys(), other.keys())):
            return False
        else:
            return True

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __len__(self) -> int:
        return sum(len(hp_name) for hp_name in self.__dict__.values())

    def __iter__(self):
        for groups in self.__dict__.values():
            yield from (groups.values())

    def __getitem__(self, key: Union[str, int]) -> _Hyperparameter:
        if isinstance(key, int):
            if not 0 <= key < len(self):
                raise IndexError("The provided key is out of bounds.")
            key = list(self.keys())[key]
        if isinstance(key, str):
            key_split = tuple(key.split("/"))
            if len(key_split) != 2:
                raise IndexError("Key string with bad syntax. Use 'param_group/param_name'.")
            group_name, hp_name = key_split
            group = getattr(self, group_name)
            return group[hp_name]
        raise ValueError("Key types supported are integer or string of syntax 'param_group/param_name'.")

    def __setitem__(self, key: Union[str, int], value: _Hyperparameter):
        if isinstance(key, int):
            if not 0 <= key < len(self):
                raise IndexError("The provided key is out of bounds.")
            key = list(self.keys())[key]
        if isinstance(key, str):
            key_split = tuple(key.split("/"))
            if len(key_split) != 2:
                raise IndexError("Key string with bad syntax. Use 'param_group/param_name'.")
            group_name, hp_name = key_split
            group = getattr(self, group_name)
            group[hp_name] = value
            return
        raise ValueError("Key types supported are integer or string of syntax 'param_group/param_name'.")

    def items(self, full_key: bool = False):
        if not full_key:
            for groups in self.__dict__.values():
                yield from (groups.items())
        else:
            yield from self.__keys_and_values()

    def _key_to_index(self, key: str) -> int:
        if not isinstance(key, str):
            raise IndexError("Key must be of type string.")
        key_split = tuple(key.split("/"))
        if len(key_split) != 2:
            raise IndexError("Key string with bad syntax. Use 'param_group/param_name'.")
        group_name, hp_name = key_split
        running_length = 0
        for _group_name in self.__dict__:
            group_dict = getattr(self, group_name)
            if group_name != _group_name:
                running_length += len(group_dict)
                continue
            if hp_name not in group_dict:
                raise KeyError(f"hyperparameter '{hp_name}' does not exist in group '{group_name}'")
            return running_length + list(group_dict.keys()).index(hp_name)
        raise KeyError(f"group '{group_name}' does not exist.")

    def keys(self):
        for group_name, group_dict in self.__dict__.items():
            for hp_name in group_dict:
                yield f"{group_name}/{hp_name}"

    def __keys_and_values(self):
        for group_name, group_dict in self.__dict__.items():
            for hp_name, hp_value in group_dict.items():
                yield (f"{group_name}/{hp_name}", hp_value)

def hyper_parameter_change_details(old_hps: Hyperparameters, new_hps: Hyperparameters) -> str:
    entries = list()
    for key in old_hps.keys():
        old_value = old_hps[key].value
        new_value = new_hps[key].value
        change = new_value - old_value
        if change != 0.0:
            change_symbol = f'+' if change > 0.0 else ''
            entries.append(f"{key}: {new_value:.2E} ({change_symbol}{change:.2E})")
        else:
            entries.append(f"{key}: {new_value:.2E}")
    return ", ".join(entries)