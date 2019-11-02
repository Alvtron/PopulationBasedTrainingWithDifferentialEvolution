import random
import numpy
from typing import TypeVar, Generic
from dataclasses import dataclass

class Hyperparameter(object):
    '''Class for creating and storing a hyper-parameter in a given, constrained search space. Provide a list of [lower bound, upper bound] as float/int/bool, or categorical elements [obj1, obj2, ..., objn]. '''
    def __init__(self, *args):
        ''' Sets the search space and sorts it, then samples a new candidate from an uniform distribution. '''
        assert args and len(list(args)) > 1, "Hyperparameter initialization needs at least two arguments."
        self.search_space = sorted(list(args))
        self.value = None
        self.sample_new()

    def __str__(self):
        return f"{self.value}, U({self.get_lower_bound()},{self.get_upper_bound()})"

    def get_lower_bound(self):
        ''' Returns the lower bounds of the hyper-parameter search space. '''
        return self.search_space[0]

    def get_upper_bound(self):
        ''' Returns the upper bounds of the hyper-parameter search space. '''
        return self.search_space[-1]

    def sample_new(self):
        ''' Samples a new candidate from an uniform distribution. '''
        # get lower and upper bound from search space
        lower_bound = self.get_lower_bound()
        upper_bound = self.get_upper_bound()
        # sample a uniform value from the search space
        if isinstance(lower_bound, float):
            self.value = random.uniform(lower_bound, upper_bound)
        elif isinstance(lower_bound, int):
            self.value = random.randint(lower_bound, upper_bound)
        elif isinstance(lower_bound, bool):
            self.value = bool(random.randbool(lower_bound, upper_bound))
        else:
            self.value = random.choice(self.search_space)

    def perturb(self, perturb_factors : tuple = (0.8, 1.2)):
        ''' Perturbs the hyper-parameter value with a given perturb factors. '''
        # get lower and upper bound from search space
        lower_bound = self.get_lower_bound()
        upper_bound = self.get_upper_bound()
        # create a random perturbation factor with the given perturb factors
        perturb_value = random.uniform(perturb_factors[0], perturb_factors[1])
        # sample a new value from the search space with the perturbation value
        if isinstance(lower_bound, float):
            self.value = float(numpy.clip(self.value * perturb_value, lower_bound, upper_bound))
        elif isinstance(lower_bound, int):
            self.value = int(numpy.clip(round(self.value * perturb_value), lower_bound, upper_bound))
        elif isinstance(lower_bound, bool):
            self.value = bool(numpy.clip(round(self.value * perturb_value), lower_bound, upper_bound))
        else:
            index = self.search_space.index(self.value)
            new_index = int(numpy.clip(round(index * perturb_value), 0, len(self.search_space) - 1))
            self.value = self.search_space[new_index]