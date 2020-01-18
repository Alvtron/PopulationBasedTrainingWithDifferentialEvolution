import random
import copy
import warnings
from functools import partial
from utils.constraint import translate, clip, reflect

class Hyperparameter(object):
    '''
    Class for creating and storing a hyperparameter in a given, constrained search space.
    '''
    def __init__(self, *args, value=None, is_categorical = False, constraint='clip'):
        ''' 
        Provide a set of [lower bound, upper bound] as float/int, or categorical elements [obj1, obj2, ..., objn].
        Make sure to set is_categorical = True if categorical values are provided.
        Sets the search space and samples a new candidate from an uniform distribution.
        '''
        if args == None:
            raise ValueError("No arguments provided.")
        self.MIN_NORM = 0.0
        self.MAX_NORM = 1.0
        if not is_categorical and not all(isinstance(arg, (float, int)) for arg in args):
            raise ValueError(f"Non-categorical hyperparameters must be of type {float} or {int}.")
        self.set_constraint(constraint)
        self.search_space = list(args)
        self.is_categorical = is_categorical
        self.__value = self.from_value(value) if value else random.uniform(self.MIN_NORM, self.MAX_NORM)

    def __translate_from_norm(self, normalized_value):
        return translate(normalized_value, self.MIN_NORM, self.MAX_NORM, self.lower_bound, self.upper_bound)
    
    def __translate_from_value(self, value):
        return translate(value, self.lower_bound, self.upper_bound, self.MIN_NORM, self.MAX_NORM)

    def set_constraint(self, constraint):
        if isinstance(constraint, str):
            if constraint == 'clip':
                self.__constrain = partial(clip, min_value=self.MIN_NORM, max_value=self.MAX_NORM)
            elif constraint == 'reflect':
                self.__constrain = partial(reflect, min_value=self.MIN_NORM, max_value=self.MAX_NORM)
            else:
                raise NotImplementedError(f"No constraint matches '{constraint}'")
        elif callable(constraint):
            self.__constrain = partial(constraint, min_value=self.MIN_NORM, max_value=self.MAX_NORM)
        else:
            raise ValueError("The provided constraint must be of type str or callable.")
        

    def __str__(self):
        return f"{self.value} U({self.lower_bound},{self.upper_bound})"

    @property
    def normalized(self):
        """Returns the normalized hyperparameter value."""
        return self.__value

    @normalized.setter
    def normalized(self, value):
        """Sets the normalized hyperparameter value."""
        self.__value = self.__constrain(value)

    @property
    def value(self):
        """Returns the representative hyperparameter value."""
        if self.__value == None:
            raise ValueError("Developer error. '__value' is None.")
        return self.from_normalized(self.__value)

    @value.setter
    def value(self, value):
        """Sets the hyperparameter value."""
        if not self.is_categorical and not(self.lower_bound <= value <= self.upper_bound):
            warnings.warn(f"The value {value} is outside the search space U({self.lower_bound}, {self.upper_bound}). The value will be constrained.")
        self.__value = self.__constrain(self.from_value(value))

    @property
    def lower_bound(self):
        ''' Returns the lower bounds of the hyper-parameter search space. If categorical, return the first search space index. '''
        return 0 if self.is_categorical else self.search_space[0]

    @property 
    def upper_bound(self):
        ''' Returns the upper bounds of the hyper-parameter search space. If categorical, return the last search space index. '''
        return len(self.search_space) - 1 if self.is_categorical else self.search_space[-1]

    def from_value(self, value):
        """Returns a normalized version of the provided value."""
        if self.is_categorical:
            assert value in self.search_space, f"The provided value {value} does not exist within the categorical search space."
            index = self.search_space.index(value)
            return self.__translate_from_value(index)
        elif isinstance(value, (int, float)):
            return self.__translate_from_value(value)
        else:
            raise Exception(f"Non-categorical hyperparameters must be of type {float} or {int}.")

    def from_normalized(self, normalized_value):
        """Returns a search space value from the provided normalized value."""
        constrained = self.__constrain(normalized_value)
        trainslated = self.__translate_from_norm(constrained)
        if self.is_categorical:
            index = int(round(trainslated))
            return self.search_space[index]
        elif isinstance(self.search_space[0], float):
            return float(trainslated)
        elif isinstance(self.search_space[0], int):
            return int(round(trainslated))
        else:
            raise Exception(f"Non-categorical hyperparameters must be of type {float} or {int}.")

    def sample_uniform(self):
        ''' Samples a new candidate from an uniform distribution bound by the lower and upper bounds. '''
        self.__value = random.uniform(self.MIN_NORM, self.MAX_NORM)
        return self.value

    def update(self, expression):
        ''' Changes the hyper-parameter value with the given expression. '''
        self.__value = float(self.__constrain(expression(self.__value)))
        return self.value

    def equal_search_space(self, other):
        return self.search_space == other.search_space and self.is_categorical == other.is_categorical

    def __add__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Addition is not supported for hyperparameters of unequal search spaces.")
            new_hp.__value = self.__constrain(new_hp.__value + other.__value)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp.__value = self.__constrain(new_hp.__value + other)
            return new_hp
        else:
            raise ValueError(f"Addition is only supported for values of type {Hyperparameter}, {float} or {int}.")

    def __sub__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Subtraction is not supported for hyperparameters of unequal search spaces.")
            new_hp.__value = self.__constrain(new_hp.__value - other.__value)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp.__value = self.__constrain(new_hp.__value - other)
            return new_hp
        else:
            raise ValueError(f"Subtraction is only supported for values of type {Hyperparameter}, {float} or {int}.")

    def __mul__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Multiplication is not supported for hyperparameters of unequal search spaces.")
            new_hp.__value = self.__constrain(new_hp.__value * other.__value)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp.__value = self.__constrain(new_hp.__value * other)
            return new_hp
        else:
            raise ValueError(f"Multiplication is only supported for values of type {Hyperparameter}, {float} or {int}.")

    def __truediv__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Divition is not supported for hyperparameters of unequal search spaces.")
            new_hp.__value = self.__constrain(new_hp.__value / other.__value)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp.__value = self.__constrain(new_hp.__value / other)
            return new_hp
        else:
            raise ValueError(f"Divition is only supported for values of type {Hyperparameter}, {float} or {int}.")

    def __pow__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Exponentiation is not supported for hyperparameters of unequal search spaces.")
            new_hp.__value = self.__constrain(new_hp.__value ** other.__value)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp.__value = self.__constrain(new_hp.__value ** other)
            return new_hp
        else:
            raise ValueError(f"Exponentiation is only supported for values of type {Hyperparameter}, {float} or {int}.")

    def __iadd__(self, other):
        if isinstance(other, Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Addition is not supported for hyperparameters of unequal search spaces.")
            self.__value = self.__constrain(self.__value + other.__value)
        elif isinstance(other, (float, int)):
            self.__value = self.__constrain(self.__value + other)
        else:
            raise ValueError(f"Addition is only supported for values of type {Hyperparameter}, {float} or {int}.")
        return self

    def __isub__(self, other):
        if isinstance(other, Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Subtraction is not supported for hyperparameters of unequal search spaces.")
            self.__value = self.__constrain(self.__value - other.__value)
        elif isinstance(other, (float, int)):
            self.__value = self.__constrain(self.__value - other)
        else:
            raise ValueError(f"Subtraction is only supported for values of type {Hyperparameter}, {float} or {int}.")
        return self

    def __imul__(self, other):
        if isinstance(other, Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Multiplication is not supported for hyperparameters of unequal search spaces.")
            self.__value = self.__constrain(self.__value * other.__value)
        elif isinstance(other, (float, int)):
            self.__value = self.__constrain(self.__value * other)
        else:
            raise ValueError(f"Multiplication is only supported for values of type {Hyperparameter}, {float} or {int}.")
        return self

    def __idiv__(self, other):
        if isinstance(other, Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Divition is not supported for hyperparameters of unequal search spaces.")
            self.__value = self.__constrain(self.__value / other.__value)
        elif isinstance(other, (float, int)):
            self.__value = self.__constrain(self.__value / other)
        else:
            raise ValueError(f"Divition is only supported for values of type {Hyperparameter}, {float} or {int}.")
        return self

    def __ipow__(self, other):
        if isinstance(other, Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Exponentiation is not supported for hyperparameters of unequal search spaces.")
            self.__value = self.__constrain(self.__value ** other.__value)
        elif isinstance(other, (float, int)):
            self.__value = clip(self.__value ** other)
        else:
            raise ValueError(f"Exponentiation is only supported for values of type {Hyperparameter}, {float} or {int}.")
        return self

    def __lt__(self, other):
        if isinstance(other, Hyperparameter) and self.search_space == other.search_space and self.is_categorical == other.is_categorical:
            return self.__value < other.__value
        else:
            raise ValueError(f"Comparison operations is supported for values of type {Hyperparameter} in equal search space.")

    def __gt__(self, other):
        if isinstance(other, Hyperparameter) and self.search_space == other.search_space and self.is_categorical == other.is_categorical:
            return self.__value > other.__value
        else:
            raise ValueError(f"Comparison operations is supported for values of type {Hyperparameter} in equal search space.")

    def __le__(self, other):
        if isinstance(other, Hyperparameter) and self.search_space == other.search_space and self.is_categorical == other.is_categorical:
            return self.__value <= other.__value
        else:
            raise ValueError(f"Comparison operations is supported for values of type {Hyperparameter} in equal search space.")

    def __ge__(self, other):
        if isinstance(other, Hyperparameter) and self.search_space == other.search_space and self.is_categorical == other.is_categorical:
            return self.__value >= other.__value
        else:
            raise ValueError(f"Comparison operations is supported for values of type {Hyperparameter} in equal search space.")

    def __eq__(self, other):
        if isinstance(other, Hyperparameter):
            return self.search_space == other.search_space and self.is_categorical == other.is_categorical and self.__value == other.__value
        else:
            raise ValueError(f"Comparison operations is supported for values of type {Hyperparameter} in equal search space.")

    def __ne__(self, other):
        if isinstance(other, Hyperparameter):
            return self.search_space != other.search_space or self.is_categorical != other.is_categorical or self.__value != other.__value
        else:
            raise ValueError(f"Comparison operations is supported for values of type {Hyperparameter} in equal search space.")

class Hyperparameters(object):
    ''' Class for storing and updating hyperparameters. '''

    def __init__(self, general_params : dict, model_params : dict, optimizer_params : dict):
        assert not general_params or all(isinstance(hyper_param, Hyperparameter) for hyper_param in general_params.values()), f"General hyperparameters can only contain {Hyperparameter} objects."
        assert not model_params or all(isinstance(hyper_param, Hyperparameter) for hyper_param in model_params.values()), f"Model hyperparameters can only contain {Hyperparameter} objects."
        assert not optimizer_params or all(isinstance(hyper_param, Hyperparameter) for hyper_param in optimizer_params.values()), f"Optimizer hyperparameters can only contain {Hyperparameter} objects."
        self.general = general_params  if general_params else {}
        self.model = model_params if model_params else {}
        self.optimizer = optimizer_params if optimizer_params else {}

    def __str__(self):
        info = []
        for name, value in self:
            info.append(f"{name}: {value}\n")
        return ''.join(info)

    def __iter__(self):
        for parameter in self.general.items():
            yield parameter
        for parameter in self.model.items():
            yield parameter
        for parameter in self.optimizer.items():
            yield parameter

    def __len__(self):
        return len(self.general) + len(self.model) + len(self.optimizer)

    def __getitem__(self, key):
        if isinstance(key, int):
            if not 0 <= key < len(self):
                raise IndexError("The provided key is out of bounds.")
            return list(self)[key][1]
        if isinstance(key, str):
            split_key = key.split("/")
            if len(split_key) != 2:
                raise IndexError("Key string with bad syntax. Use 'param_group/param_name'.")
            group = getattr(self, split_key[0])
            return group[split_key[1]]
        raise ValueError("Key types supported are int and str of syntax 'param_group/param_name'.")

    def __setitem__(self, key, value):
        if not 0 <= key < len(self):
            raise IndexError("The provided key is out of bounds.")
        if key < len(self.general):
            param_name = list(self.general)[key]
            self.general[param_name] = value
        elif key < len(self.general) + len(self.model):
            param_name = list(self.model)[key - len(self.general)]
            self.model[param_name] = value
        else:
            param_name = list(self.optimizer)[key - len(self.general) - len(self.model)]
            self.optimizer[param_name] = value

    def parameters(self):
        return [i[1] for i in self]

    def names(self):
        return [i[0] for i in self]

    def keys(self):
        general_paths = [f"general/{parameter}" for parameter in self.general]
        model_paths = [f"model/{parameter}" for parameter in self.model]
        optimizer_paths = [f"optimizer/{parameter}" for parameter in self.optimizer]
        return general_paths + model_paths + optimizer_paths

    def values(self):
        return [i[1].value for i in self]
    
    def normalized(self):
        return [i[1].normalized for i in self]

    def set(self, list):
        length = len(self)
        if len(list) != length:
            raise ValueError("The provided hyperparameter list must be of same length as this configuration.")
        for index in range(length):
            self[index] += list[index]

    def get_general_value_dict(self):
        return {param_name:param.value for param_name, param in self.general.items()}

    def get_model_value_dict(self):
        return {param_name:param.value for param_name, param in self.model.items()}

    def get_optimizer_value_dict(self):
        return {param_name:param.value for param_name, param in self.optimizer.items()}

    def equal_search_space(self, other):
        # Check if each hyperparameter type is of same length
        if len(self.general) != len(other.general):
            return False
        if len(self.model) != len(other.model):
            return False
        if len(self.optimizer) != len(other.optimizer):
            return False
        # Check if hyperparameter names are equal
        if self.general.keys() != other.general.keys():
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