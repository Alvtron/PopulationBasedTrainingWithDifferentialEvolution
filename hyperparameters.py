import random
import copy

def translate(value, left_min, left_max, right_min, right_max):
    # Calculate the span of each range
    left_span = left_max - left_min
    right_span = right_max - right_min
    # normalize the value from the left range into a float between 0 and 1
    value_normalized = float(value - left_min) / float(left_span)
    # Convert the normalize value range into a value in the right range.
    return right_min + (value_normalized * right_span)

def clip(value, min_value, max_value):
    if value <= min_value:
        return min_value
    elif value >= max_value:
        return max_value
    else:
        return value

def translate_and_clip(value, left_min, left_max, right_min, right_max):
    translated_value = translate(value, left_min, left_max, right_min, right_max)
    clipped_value = clip(translated_value, right_min, right_max)
    return clipped_value

class Hyperparameter(object):
    ''' Class for creating and storing a hyperparameter in a given, constrained search space. '''

    def __init__(self, *args, is_categorical = False):
        ''' Provide a set of [lower bound, upper bound] as float/int, or categorical elements [obj1, obj2, ..., objn]. Make sure to set is_categorical = True if categorical values are provided. Sets the search space and sorts it, then samples a new candidate from an uniform distribution. '''
        args = args if args and len(list(args)) > 1 else (0.0, 1.0)
        assert is_categorical or isinstance(args[0], (float, int)), f"Non-categorical hyperparameters must be of type {float} or {int}."
        self.search_space = list(args)
        self.is_categorical = is_categorical
        self.__value = random.uniform(0.0, 1.0)

    def __str__(self):
        return f"{self.value()} U({self.get_lower_bound()},{self.get_upper_bound()})"

    def normalized(self):
        """Returns the normalized hyperparameter value."""
        return self.__value

    def value(self):
        """Returns the representative hyperparameter value."""
        if self.__value == None:
            raise ValueError("Developer error. '__value' is None.")
        return self.get_value(self.__value)

    def get_normalized_value(self, value):
        """Returns a normalized version of the provided hyperparameter value."""
        if self.is_categorical:
            assert value in self.search_space, f"The provided value {value} does not exist within the categorical search space."
            index = self.search_space.index(value)
            return translate(index, 0, len(self.search_space) - 1, 0.0, 1.0)
        elif isinstance(value, (int, float)):
            return translate(value, self.get_lower_bound(), self.get_upper_bound(), 0.0, 1.0)
        else:
            raise Exception(f"Non-categorical hyperparameters must be of type {float} or {int}.")

    def get_value(self, normalized_value):
        """Returns a normalized version of the provided hyperparameter value."""
        if self.is_categorical:
            index = int(round(translate_and_clip(
                normalized_value,
                0.0, 1.0,
                self.get_lower_bound(),
                self.get_upper_bound())))
            return self.search_space[index]
        elif isinstance(self.search_space[0], float):
            return float(translate_and_clip(
                normalized_value,
                0.0, 1.0,
                self.get_lower_bound(),
                self.get_upper_bound()))
        elif isinstance(self.search_space[0], int):
            return int(round(translate_and_clip(
                normalized_value,
                0.0, 1.0,
                self.get_lower_bound(),
                self.get_upper_bound())))
        else:
            raise Exception(f"Non-categorical hyperparameters must be of type {float} or {int}.")

    def set_normalized_value(self, value):
        """Sets the normalized hyperparameter value."""
        assert 0.0 <= value <= 1.0, "The normalized value must be between 0.0 and 1.0."
        self.__value = clip(value, 0.0, 1.0)

    def set_value(self, value):
        """Sets the hyperparameter value."""
        if not self.is_categorical:
            assert self.get_lower_bound() <= value <= self.get_upper_bound(), "The value must be between {self.get_lower_bound()} and {self.get_upper_bound()}."
        normalized_value = self.get_normalized_value(value)
        self.set_normalized_value(normalized_value)

    def get_lower_bound(self):
        ''' Returns the lower bounds of the hyper-parameter search space. If categorical, return the first search space index. '''
        return 0 if self.is_categorical else self.search_space[0]

    def get_upper_bound(self):
        ''' Returns the upper bounds of the hyper-parameter search space. If categorical, return the last search space index. '''
        return len(self.search_space) - 1 if self.is_categorical else self.search_space[-1]

    def sample_uniform(self):
        ''' Samples a new candidate from an uniform distribution bound by the lower and upper bounds. '''
        self.__value = random.uniform(0.0, 1.0)
        return self.value()

    def update(self, expression):
        ''' Changes the hyper-parameter value with the given expression. '''
        self.__value = float(clip(expression(self.__value), 0.0, 1.0))
        return self.value()

    def equal_search_space(self, other):
        return self.search_space == other.search_space and self.is_categorical == other.is_categorical

    def __add__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Addition is not supported for hyperparameters of unequal search spaces.")
            new_hp.__value = clip(new_hp.__value + other.__value, 0.0, 1.0)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp.__value = clip(new_hp.__value + other, 0.0, 1.0)
            return new_hp
        else:
            raise ValueError(f"Addition is only supported for values of type {Hyperparameter}, {float} or {int}.")

    def __sub__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Subtraction is not supported for hyperparameters of unequal search spaces.")
            new_hp.__value = clip(new_hp.__value - other.__value, 0.0, 1.0)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp.__value = clip(new_hp.__value - other, 0.0, 1.0)
            return new_hp
        else:
            raise ValueError(f"Subtraction is only supported for values of type {Hyperparameter}, {float} or {int}.")

    def __mul__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Multiplication is not supported for hyperparameters of unequal search spaces.")
            new_hp.__value = clip(new_hp.__value * other.__value, 0.0, 1.0)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp.__value = clip(new_hp.__value * other, 0.0, 1.0)
            return new_hp
        else:
            raise ValueError(f"Multiplication is only supported for values of type {Hyperparameter}, {float} or {int}.")

    def __truediv__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Divition is not supported for hyperparameters of unequal search spaces.")
            new_hp.__value = clip(new_hp.__value / other.__value, 0.0, 1.0)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp.__value = clip(new_hp.__value / other, 0.0, 1.0)
            return new_hp
        else:
            raise ValueError(f"Divition is only supported for values of type {Hyperparameter}, {float} or {int}.")

    def __pow__(self, other):
        new_hp = copy.deepcopy(self)
        if isinstance(other, Hyperparameter):
            if not new_hp.equal_search_space(other):
                raise ValueError("Exponentiation is not supported for hyperparameters of unequal search spaces.")
            new_hp.__value = clip(new_hp.__value ** other.__value, 0.0, 1.0)
            return new_hp
        elif isinstance(other, (float, int)):
            new_hp.__value = clip(new_hp.__value ** other, 0.0, 1.0)
            return new_hp
        else:
            raise ValueError(f"Exponentiation is only supported for values of type {Hyperparameter}, {float} or {int}.")

    def __iadd__(self, other):
        if isinstance(other, Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Addition is not supported for hyperparameters of unequal search spaces.")
            self.__value = clip(self.__value + other.__value, 0.0, 1.0)
        elif isinstance(other, (float, int)):
            self.__value = clip(self.__value + other, 0.0, 1.0)
        else:
            raise ValueError(f"Addition is only supported for values of type {Hyperparameter}, {float} or {int}.")
        return self

    def __isub__(self, other):
        if isinstance(other, Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Subtraction is not supported for hyperparameters of unequal search spaces.")
            self.__value = clip(self.__value - other.__value, 0.0, 1.0)
        elif isinstance(other, (float, int)):
            self.__value = clip(self.__value - other, 0.0, 1.0)
        else:
            raise ValueError(f"Subtraction is only supported for values of type {Hyperparameter}, {float} or {int}.")
        return self

    def __imul__(self, other):
        if isinstance(other, Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Multiplication is not supported for hyperparameters of unequal search spaces.")
            self.__value = clip(self.__value * other.__value, 0.0, 1.0)
        elif isinstance(other, (float, int)):
            self.__value = clip(self.__value * other, 0.0, 1.0)
        else:
            raise ValueError(f"Multiplication is only supported for values of type {Hyperparameter}, {float} or {int}.")
        return self

    def __idiv__(self, other):
        if isinstance(other, Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Divition is not supported for hyperparameters of unequal search spaces.")
            self.__value = clip(self.__value / other.__value, 0.0, 1.0)
        elif isinstance(other, (float, int)):
            self.__value = clip(self.__value / other, 0.0, 1.0)
        else:
            raise ValueError(f"Divition is only supported for values of type {Hyperparameter}, {float} or {int}.")
        return self

    def __ipow__(self, other):
        if isinstance(other, Hyperparameter):
            if not self.equal_search_space(other):
                raise ValueError("Exponentiation is not supported for hyperparameters of unequal search spaces.")
            self.__value = clip(self.__value ** other.__value, 0.0, 1.0)
        elif isinstance(other, (float, int)):
            self.__value = clip(self.__value ** other, 0.0, 1.0)
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
        if not 0 <= key < len(self):
            raise IndexError("The provided key is out of bounds.")
        return list(self)[key][1]

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

    def parameter_names(self):
        return [i[0] for i in self]

    def values(self):
        return [i[1].value() for i in self]
    
    def normalized(self):
        return [i[1].normalized() for i in self]

    def set(self, list):
        length = len(self)
        if len(list) != length:
            raise ValueError("The provided hyperparameter list must be of same length as this configuration.")
        for index in range(length):
            self[index] += list[index]

    def get_general_value_dict(self):
        return {param_name:param.value() for param_name, param in self.general.items()}

    def get_model_value_dict(self):
        return {param_name:param.value() for param_name, param in self.model.items()}

    def get_optimizer_value_dict(self):
        return {param_name:param.value() for param_name, param in self.optimizer.items()}

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