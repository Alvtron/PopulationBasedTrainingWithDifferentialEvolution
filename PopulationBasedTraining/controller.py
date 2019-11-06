import abc
import math
import operator
import random
from utils import get_datetime_string
from database import Checkpoint, Database
from hyperparameters import Hyperparameter

class Controller(abc.ABC):
    @abc.abstractmethod
    def prepare(self, hyperparameters):
        pass

    @abc.abstractmethod
    def evolve(self, member, database):
        pass

    @abc.abstractmethod
    def is_ready(self, member, database):
        pass

    @abc.abstractmethod
    def is_finished(self, member, database):
        pass

class ExploitAndExplore(Controller):
    """ A general, modifiable implementation of PBTs exploitation and exploration method. """
    def __init__(self, exploit_factor = 0.4, explore_factors = (1.2, 0.8), frequency = 5, end_criteria = {'epoch': 20, 'score': 99.9}):
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors
        self.frequency = frequency
        self.end_criteria = end_criteria

    def prepare(self, hyperparameters):
        # preparing optimizer
        for hyperparameter in hyperparameters['optimizer'].values():
            hyperparameter.sample_uniform()
        # preparing batch_size
        if hyperparameters['batch_size']:
            hyperparameters['batch_size'].sample_uniform()

    def evolve(self, member, database):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        population = database.get_latest()
        # sort members
        population.sort(key=lambda m: m.score, reverse=True)
        # set number of elitists
        n_elitists = math.floor(len(population) * self.exploit_factor)
        if n_elitists > 0 and all(c.id != member.id for c in population[:n_elitists]):
            # exploit
            self.exploit(member, population[:n_elitists])
        # explore
        self.explore(member)

    def is_ready(self, member, database):
        """True every n-th epoch."""
        return member.epoch % self.frequency == 0

    def is_finished(self, member, database):
        """ True if a given number of epochs have passed or if the score reaches above 99%. """
        if 'epoch' in self.end_criteria and member.epoch > self.end_criteria['epoch']:
            # the number of epochs is above the given treshold
            return True
        all_members = database.to_list()
        if not all_members:
            # database is currently empty
            return False
        best_score = max(c.score for c in all_members)
        if 'score' in self.end_criteria and best_score > self.end_criteria['score']:
            # score is above the given treshold
            return True
        return False
      
    def exploit(self, member, population):
        elitist = random.choice(population)
        print(f"{get_datetime_string()} - epoch {member.epoch} - m{member.id}*: exploiting w{elitist.id}...")
        member.update(elitist)

    def explore(self, member):
        print(f"{get_datetime_string()} - epoch {member.epoch} - m{member.id}*: exploring...")
        # exploring optimizer
        for hyperparameter_name, hyperparameter in member.hyperparameters['optimizer'].items():
            old_value = hyperparameter.get_value()
            hyperparameter *= random.choice(self.explore_factors)
            new_value = hyperparameter.get_value()
            print(f"{hyperparameter_name}: {old_value:.2f} --> {new_value:.2f}")
        # exploring batch_size
        if member.hyperparameters['batch_size']:
            old_value = member.hyperparameters['batch_size'].get_value()
            member.hyperparameters['batch_size'] *= random.choice(self.explore_factors)
            new_value = member.hyperparameters['batch_size'].get_value()
            print(f"batch_size: {old_value:.2f} --> {new_value:.2f}")