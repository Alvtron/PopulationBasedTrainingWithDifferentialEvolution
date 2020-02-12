import random
import copy
import warnings
from functools import partial
from typing import Dict
from abc import abstractmethod

from .utils.constraint import translate, clip, reflect

class InvalidSearchSpaceException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class _Hyperparameter(object):
    '''
    Class for creating and storing a hyperparameter in a given, constrained search space.
    '''
    def __init__(self, *args, value=None, constraint='clip'):
        ''' 
        Provide a set of [lower bound, upper bound] as float/int, or categorical elements [obj1, obj2, ..., objn].
        Sets the search space and samples a new candidate from an uniform distribution.
        '''
        if args == None:
            raise ValueError("No arguments provided.")
        self.MIN_NORM = 0.0
        self.MAX_NORM = 1.0
        self.set_constraint(constraint)
        self.search_space = list(args)
        self._normalized = self.from_value(value) if value is not None else random.uniform(self.MIN_NORM, self.MAX_NORM)

    def _translate_from_norm(self, normalized_value) -> float:
        return translate(normalized_value, self.MIN_NORM, self.MAX_NORM, self.lower_bound, self.upper_bound)
    
    def _translate_from_value(self, value) -> float:
        return translate(value, self.lower_bound, self.upper_bound, self.MIN_NORM, self.MAX_NORM)

    def set_constraint(self, constraint):
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
        return f"v: {self.value}, n: {self.normalized:.3f} U({self.lower_bound},{self.upper_bound})"

    @property
    def normalized(self) -> float:
        """Returns the normalized hyperparameter value."""
        return self._normalized

    @normalized.setter
    def normalized(self, value) -> float:
        """Sets the normalized hyperparameter value."""
        self._normalized = self._constrain(value)
    
    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError

    @value.setter
    @abstractmethod
    def value(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def lower_bound(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def upper_bound(self):
        raise NotImplementedError

    @abstractmethod
    def from_value(self, value):
        raise NotImplementedError

    @abstractmethod
    def from_normalized(self, normalized_value):
        raise NotImplementedError

    def sample_uniform(self):
        ''' Samples a new candidate from an uniform distribution bound by the lower and upper bounds. '''
        self._normalized = random.uniform(self.MIN_NORM, self.MAX_NORM)
        return self.value

    def update(self, expression):
        ''' Changes the hyper-parameter value with the given expression. '''
        self._normalized = float(self._constrain(expression(self._normalized)))
        return self.value

    def equal_search_space(self, other) -> bool:
        """Return true if the search space is equal."""
        return isinstance(other, _Hyperparameter) and self.search_space == other.search_space

    def __add__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, _Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Addition is not supported for hyperparameters of unequal search spaces.")
            new_hp._normalized = self._constrain(new_hp._normalized + other._normalized)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp._normalized = self._constrain(new_hp._normalized + other)
            return new_hp
        else:
            raise ValueError(f"Addition is only supported for values of type {_Hyperparameter}, {float} or {int}.")

    def __sub__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, _Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Subtraction is not supported for hyperparameters of unequal search spaces.")
            new_hp._normalized = self._constrain(new_hp._normalized - other._normalized)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp._normalized = self._constrain(new_hp._normalized - other)
            return new_hp
        else:
            raise ValueError(f"Subtraction is only supported for values of type {_Hyperparameter}, {float} or {int}.")

    def __mul__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, _Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Multiplication is not supported for hyperparameters of unequal search spaces.")
            new_hp._normalized = self._constrain(new_hp._normalized * other._normalized)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp._normalized = self._constrain(new_hp._normalized * other)
            return new_hp
        else:
            raise ValueError(f"Multiplication is only supported for values of type {_Hyperparameter}, {float} or {int}.")

    def __truediv__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, _Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Divition is not supported for hyperparameters of unequal search spaces.")
            new_hp._normalized = self._constrain(new_hp._normalized / other._normalized)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp._normalized = self._constrain(new_hp._normalized / other)
            return new_hp
        else:
            raise ValueError(f"Divition is only supported for values of type {_Hyperparameter}, {float} or {int}.")

    def __pow__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, _Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Exponentiation is not supported for hyperparameters of unequal search spaces.")
            new_hp._normalized = self._constrain(new_hp._normalized ** other._normalized)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp._normalized = self._constrain(new_hp._normalized ** other)
            return new_hp
        else:
            raise ValueError(f"Exponentiation is only supported for values of type {_Hyperparameter}, {float} or {int}.")

    def __iadd__(self, other):
        if isinstance(other, _Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Addition is not supported for hyperparameters of unequal search spaces.")
            self._normalized = self._constrain(self._normalized + other._normalized)
        elif isinstance(other, (float, int)):
            self._normalized = self._constrain(self._normalized + other)
        else:
            raise ValueError(f"Addition is only supported for values of type {_Hyperparameter}, {float} or {int}.")
        return self

    def __isub__(self, other):
        if isinstance(other, _Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Subtraction is not supported for hyperparameters of unequal search spaces.")
            self._normalized = self._constrain(self._normalized - other._normalized)
        elif isinstance(other, (float, int)):
            self._normalized = self._constrain(self._normalized - other)
        else:
            raise ValueError(f"Subtraction is only supported for values of type {_Hyperparameter}, {float} or {int}.")
        return self

    def __imul__(self, other):
        if isinstance(other, _Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Multiplication is not supported for hyperparameters of unequal search spaces.")
            self._normalized = self._constrain(self._normalized * other._normalized)
        elif isinstance(other, (float, int)):
            self._normalized = self._constrain(self._normalized * other)
        else:
            raise ValueError(f"Multiplication is only supported for values of type {_Hyperparameter}, {float} or {int}.")
        return self

    def __idiv__(self, other):
        if isinstance(other, _Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Divition is not supported for hyperparameters of unequal search spaces.")
            self._normalized = self._constrain(self._normalized / other._normalized)
        elif isinstance(other, (float, int)):
            self._normalized = self._constrain(self._normalized / other)
        else:
            raise ValueError(f"Divition is only supported for values of type {_Hyperparameter}, {float} or {int}.")
        return self

    def __ipow__(self, other):
        if isinstance(other, _Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Exponentiation is not supported for hyperparameters of unequal search spaces.")
            self._normalized = self._constrain(self._normalized ** other._normalized)
        elif isinstance(other, (float, int)):
            self._normalized = clip(self._normalized ** other)
        else:
            raise ValueError(f"Exponentiation is only supported for values of type {_Hyperparameter}, {float} or {int}.")
        return self

    def __lt__(self, other):
        if isinstance(other, _Hyperparameter) and self.search_space == other.search_space:
            return self._normalized < other._normalized
        else:
            raise ValueError(f"Comparison operations is supported for values of type {_Hyperparameter} in equal search space.")

    def __gt__(self, other):
        if isinstance(other, _Hyperparameter) and self.search_space == other.search_space:
            return self._normalized > other._normalized
        else:
            raise ValueError(f"Comparison operations is supported for values of type {_Hyperparameter} in equal search space.")

    def __le__(self, other):
        if isinstance(other, _Hyperparameter) and self.search_space == other.search_space:
            return self._normalized <= other._normalized
        else:
            raise ValueError(f"Comparison operations is supported for values of type {_Hyperparameter} in equal search space.")

    def __ge__(self, other):
        if isinstance(other, _Hyperparameter) and self.search_space == other.search_space:
            return self._normalized >= other._normalized
        else:
            raise ValueError(f"Comparison operations is supported for values of type {_Hyperparameter} in equal search space.")

    def __eq__(self, other):
        if isinstance(other, _Hyperparameter):
            return self.search_space == other.search_space and self._normalized == other._normalized
        else:
            raise ValueError(f"Comparison operations is supported for values of type {_Hyperparameter} in equal search space.")

    def __ne__(self, other):
        if isinstance(other, _Hyperparameter):
            return self.search_space != other.search_space or self._normalized != other._normalized
        else:
            raise ValueError(f"Comparison operations is supported for values of type {_Hyperparameter} in equal search space.")

class ContiniousHyperparameter(_Hyperparameter):
    '''
    Class for creating and storing a hyperparameter in a given, constrained search space.
    '''
    def __init__(self, minimum, maximum, value=None, constraint='clip'):
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
    def value(self):
        """Returns the representative hyperparameter value."""
        if self._normalized == None:
            raise ValueError("Developer error. '_normalized' is None.")
        return self.from_normalized(self._normalized)

    @value.setter
    def value(self, value):
        """Sets the hyperparameter value."""
        if not(self.lower_bound <= value <= self.upper_bound):
            warnings.warn(f"The value {value} is outside the search space U({self.lower_bound}, {self.upper_bound}). The value will be constrained.")
        self._normalized = self._constrain(self.from_value(value))

    @property
    def lower_bound(self):
        ''' Returns the lower bounds of the hyper-parameter search space. If categorical, return the first search space index. '''
        return self.search_space[0]

    @property 
    def upper_bound(self):
        ''' Returns the upper bounds of the hyper-parameter search space. If categorical, return the last search space index. '''
        return self.search_space[-1]

    def from_value(self, value):
        """Returns a normalized version of the provided value."""
        if isinstance(value, (int, float)):
            return self._translate_from_value(value)
        else:
            raise Exception(f"Non-categorical hyperparameters must be of type {float} or {int}.")

    def from_normalized(self, normalized_value):
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
    def __init__(self, *search_space, value=None, constraint='clip'):
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
    def value(self):
        """Returns the representative hyperparameter value."""
        if self._normalized == None:
            raise ValueError("Developer error. '_normalized' is None.")
        return self.from_normalized(self._normalized)

    @value.setter
    def value(self, value):
        """Sets the hyperparameter value."""
        if value not in self.search_space:
            raise ValueError("The provided value must be present in the categorical search space.")
        self._normalized = self._constrain(self.from_value(value))

    @property
    def lower_bound(self):
        ''' Returns the lower bounds of the hyper-parameter search space. If categorical, return the first search space index. '''
        return 0

    @property 
    def upper_bound(self):
        ''' Returns the upper bounds of the hyper-parameter search space. If categorical, return the last search space index. '''
        return len(self.search_space) - 1

    def from_value(self, value):
        """Returns a normalized version of the provided value."""
        assert value in self.search_space, f"The provided value {value} does not exist within the categorical search space."
        index = self.search_space.index(value)
        return self._translate_from_value(index)

    def from_normalized(self, normalized_value):
        """Returns a search space value from the provided normalized value."""
        constrained = self._constrain(normalized_value)
        trainslated = self._translate_from_norm(constrained)
        index = int(round(trainslated))
        return self.search_space[index]

    def equal_search_space(self, other):
        return isinstance(other, DiscreteHyperparameter) and super().equal_search_space(other)

    def __lt__(self, other):
        if not isinstance(other, DiscreteHyperparameter):
            raise ValueError(f"Comparison operations are only supported for values of type {DiscreteHyperparameter}.")
        return super().__lt__(other)

    def __gt__(self, other):
        if not isinstance(other, DiscreteHyperparameter):
            raise ValueError(f"Comparison operations are only supported for values of type {DiscreteHyperparameter}.")
        return super().__gt__(other)

    def __le__(self, other):
        if not isinstance(other, DiscreteHyperparameter):
            raise ValueError(f"Comparison operations are only supported for values of type {DiscreteHyperparameter}.")
        return super().__le__(other)

    def __ge__(self, other):
        if not isinstance(other, DiscreteHyperparameter):
            raise ValueError(f"Comparison operations are only supported for values of type {DiscreteHyperparameter}.")
        return super().__ge__(other)

    def __eq__(self, other):
        if not isinstance(other, DiscreteHyperparameter):
            raise ValueError(f"Comparison operations are only supported for values of type {DiscreteHyperparameter}.")
        return super().__eq__(other)

    def __ne__(self, other):
        if not isinstance(other, DiscreteHyperparameter):
            raise ValueError(f"Comparison operations are only supported for values of type {DiscreteHyperparameter}.")
        return super().__ne__(other)

class Hyperparameters(object):
    ''' Class for storing and updating hyperparameters. '''

    def __init__(self, augment_params : Dict[str, _Hyperparameter], model_params : Dict[str, _Hyperparameter], optimizer_params : Dict[str, _Hyperparameter]):
        if augment_params and not all(isinstance(hyper_param, _Hyperparameter) for hyper_param in augment_params.values()):
            raise TypeError(f"General hyperparameters can only contain {_Hyperparameter} objects.")
        if model_params and not all(isinstance(hyper_param, _Hyperparameter) for hyper_param in model_params.values()):
            raise TypeError(f"Model hyperparameters can only contain {_Hyperparameter} objects.")
        if optimizer_params and not all(isinstance(hyper_param, _Hyperparameter) for hyper_param in optimizer_params.values()):
            raise TypeError(f"Optimizer hyperparameters can only contain {_Hyperparameter} objects.")
        self.augment : Dict[str, _Hyperparameter] = augment_params  if augment_params else {}
        self.model : Dict[str, _Hyperparameter] = model_params if model_params else {}
        self.optimizer : Dict[str, _Hyperparameter] = optimizer_params if optimizer_params else {}

    def __str__(self):
        info = []
        for name, value in self:
            info.append(f"{name}: {value}\n")
        return ''.join(info)

    def __iter__(self):
        for parameter in self.augment.items():
            yield parameter
        for parameter in self.model.items():
            yield parameter
        for parameter in self.optimizer.items():
            yield parameter

    def __len__(self):
        return len(self.augment) + len(self.model) + len(self.optimizer)

    def __getitem__(self, key):
        if isinstance(key, int):
            if not 0 <= key < len(self):
                raise IndexError("The provided key is out of bounds.")
            return list(self)[key][1]
        if isinstance(key, str):
            split_key = key.split("/")
            if len(split_key) != 2:
                raise IndexError("Key string with bad syntax. Use 'param_group/param_name'.")
            split_key[0] = "augment" if split_key[0] == "general" else split_key[0]
            group = getattr(self, split_key[0])
            return group[split_key[1]]
        raise ValueError("Key types supported are int and str of syntax 'param_group/param_name'.")

    def __setitem__(self, key, value):
        if not 0 <= key < len(self):
            raise IndexError("The provided key is out of bounds.")
        if key < len(self.augment):
            param_name = list(self.augment)[key]
            self.augment[param_name] = value
        elif key < len(self.augment) + len(self.model):
            param_name = list(self.model)[key - len(self.augment)]
            self.model[param_name] = value
        else:
            param_name = list(self.optimizer)[key - len(self.augment) - len(self.model)]
            self.optimizer[param_name] = value

    def parameters(self):
        return (i[1] for i in self)

    def names(self):
        return (i[0] for i in self)

    def keys(self):
        general_paths = [f"augment/{parameter}" for parameter in self.augment]
        model_paths = [f"model/{parameter}" for parameter in self.model]
        optimizer_paths = [f"optimizer/{parameter}" for parameter in self.optimizer]
        return general_paths + model_paths + optimizer_paths

    def values(self):
        return (hp_object.value for _, hp_object in self)
    
    def normalized(self):
        return (hp_object.normalized for _, hp_object in self)

    def categorical(self):
        return {hp_name: hp_object for hp_name, hp_object in self if isinstance(hp_object, DiscreteHyperparameter)}

    def non_categorical(self):
        return {hp_name: hp_object for hp_name, hp_object in self if not isinstance(hp_object, DiscreteHyperparameter)}

    def set(self, list):
        length = len(self)
        if len(list) != length:
            raise ValueError("The provided hyperparameter list must be of same length as this configuration.")
        for index in range(length):
            self[index] += list[index]

    def get_augment_value_dict(self):
        return {name:param.value for name, param in self.augment.items()}

    def get_model_value_dict(self):
        return {name:param.value for name, param in self.model.items()}

    def get_optimizer_value_dict(self):
        return {name:param.value for name, param in self.optimizer.items()}

    def equal_search_space(self, other):
        # Check if each hyperparameter type is of same length
        if len(self.augment) != len(other.augment):
            return False
        if len(self.model) != len(other.model):
            return False
        if len(self.optimizer) != len(other.optimizer):
            return False
        # Check if hyperparameter names are equal
        if self.augment.keys() != other.augment.keys():
            return False
        if self.model.keys() != other.model.keys():
            return False
        if self.optimizer.keys() != other.optimizer.keys():
            return False
        # Check if every hyperparameter dimension is of equal search space.
        for index in range(len(self)):
            if not self[index].equal_search_space(other[index]):
                return False
        return True

    def __add__(self, other):
        new_hp = copy.deepcopy(self)
        length = len(self)
        if isinstance(other, Hyperparameters):
            if not new_hp.equal_search_space(other):
                raise ValueError("Addition is not supported for hyperparameter configurations of unequal search spaces.")
            for index in range(length):
                new_hp[index] = self[index] + other[index]
            return new_hp
        elif isinstance(other, (float, int)):
            for index in range(length):
                new_hp[index] = self[index] + other
            return new_hp
        else:
            raise ValueError(f"Addition is supported for values of type {Hyperparameters}, {float} or {int}.")

    def __sub__(self, other):
        new_hp = copy.deepcopy(self)
        length = len(self)
        if isinstance(other, Hyperparameters):
            if not new_hp.equal_search_space(other):
                raise ValueError("Subtraction is not supported for hyperparameter configurations of unequal search spaces.")
            for index in range(length):
                new_hp[index] = self[index] - other[index]
            return new_hp
        elif isinstance(other, (float, int)):
            for index in range(length):
                new_hp[index] = self[index] - other
            return new_hp
        else:
            raise ValueError(f"Subtraction is supported for values of type {Hyperparameters}, {float} or {int}.")

    def __mul__(self, other):
        new_hp = copy.deepcopy(self)
        length = len(self)
        if isinstance(other, Hyperparameters):
            if not new_hp.equal_search_space(other):
                raise ValueError("Multiplication is not supported for hyperparameter configurations of unequal search spaces.")
            for index in range(length):
                new_hp[index] = self[index] * other[index]
            return new_hp
        elif isinstance(other, (float, int)):
            for index in range(length):
                new_hp[index] = self[index] * other
            return new_hp
        else:
            raise ValueError(f"Multiplication is supported for values of type {Hyperparameters}, {float} or {int}.")

    def __truediv__(self, other):
        new_hp = copy.deepcopy(self)
        length = len(self)
        if isinstance(other, Hyperparameters):
            if not new_hp.equal_search_space(other):
                raise ValueError("Divition is not supported for hyperparameter configurations of unequal search spaces.")
            for index in range(length):
                new_hp[index] = self[index] / other[index]
            return new_hp
        elif isinstance(other, (float, int)):
            for index in range(length):
                new_hp[index] = self[index] / other
            return new_hp
        else:
            raise ValueError(f"Divition is supported for values of type {Hyperparameters}, {float} or {int}.")

    def __pow__(self, other):
        new_hp = copy.deepcopy(self)
        length = len(self)
        if isinstance(other, Hyperparameters):
            if not new_hp.equal_search_space(other):
                raise ValueError("Exponentiation is not supported for hyperparameter configurations of unequal search spaces.")
            for index in range(length):
                new_hp[index] = self[index] ** other[index]
            return new_hp
        elif isinstance(other, (float, int)):
            for index in range(length):
                new_hp[index] = self[index] ** other
            return new_hp
        else:
            raise ValueError(f"Exponentiation is supported for values of type {Hyperparameters}, {float} or {int}.")

    def __iadd__(self, other):
        length = len(self)
        if isinstance(other, Hyperparameters):
            if not self.equal_search_space(other):
                raise ValueError("Addition is not supported for hyperparameter configurations of unequal search spaces.")
            for index in range(length):
                self[index] += other[index]
            return self
        elif isinstance(other, (float, int)):
            for index in range(length):
                self[index] += other
            return self
        else:
            raise ValueError(f"Addition is supported for values of type {Hyperparameters}, {float} or {int}.")

    def __isub__(self, other):
        length = len(self)
        if isinstance(other, Hyperparameters):
            if not self.equal_search_space(other):
                raise ValueError("Subtraction is not supported for hyperparameter configurations of unequal search spaces.")
            for index in range(length):
                self[index] -= other[index]
            return self
        elif isinstance(other, (float, int)):
            for index in range(length):
                self[index] -= other
            return self
        else:
            raise ValueError(f"Subtraction is supported for values of type {Hyperparameters}, {float} or {int}.")

    def __imul__(self, other):
        length = len(self)
        if isinstance(other, Hyperparameters):
            if not self.equal_search_space(other):
                raise ValueError("Multiplication is not supported for hyperparameter configurations of unequal search spaces.")
            for index in range(length):
                self[index] *= other[index]
            return self
        elif isinstance(other, (float, int)):
            for index in range(length):
                self[index] *= other
            return self
        else:
            raise ValueError(f"Multiplication is supported for values of type {Hyperparameters}, {float} or {int}.")

    def __idiv__(self, other):
        length = len(self)
        if isinstance(other, Hyperparameters):
            if not self.equal_search_space(other):
                raise ValueError("Divition is not supported for hyperparameter configurations of unequal search spaces.")
            for index in range(length):
                self[index] /= other[index]
            return self
        elif isinstance(other, (float, int)):
            for index in range(length):
                self[index] /= other
            return self
        else:
            raise ValueError(f"Divition is supported for values of type {Hyperparameters}, {float} or {int}.")

    def __ipow__(self, other):
        length = len(self)
        if isinstance(other, Hyperparameters):
            if not self.equal_search_space(other):
                raise ValueError("Exponentiation is not supported for hyperparameter configurations of unequal search spaces.")
            for index in range(length):
                self[index] **= other[index]
            return self
        elif isinstance(other, (float, int)):
            for index in range(length):
                self[index] **= other
            return self
        else:
            raise ValueError(f"Exponentiation is supported for values of type {Hyperparameters}, {float} or {int}.")

    def __eq__(self, other):
        if isinstance(other, Hyperparameters):
            for (param_name_1, param_value_1), (param_name_2, param_value_2) in zip(self, other):
                if param_value_1 != param_value_2 or param_name_1 != param_name_2:
                    return False
            return True
        else:
            raise ValueError(f"Comparison is supported for values of type {Hyperparameters}.")

    def __ne__(self, other):
        if isinstance(other, Hyperparameters):
            for (param_name_1, param_value_1), (param_name_2, param_value_2) in zip(self, other):
                if param_value_1 != param_value_2 or param_name_1 != param_name_2:
                    return True
            return False
        else:
            raise ValueError(f"Comparison is supported for values of type {Hyperparameters}.")