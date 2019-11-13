import abc
import math
import operator
import random
from database import Checkpoint, Database
from hyperparameters import Hyperparameter

class Controller(abc.ABC):
    @abc.abstractmethod
    def prepare(self, hyperparameters, logger):
        pass

    @abc.abstractmethod
    def evolve(self, member, database, logger):
        pass

    @abc.abstractmethod
    def is_ready(self, member, database):
        pass

    @abc.abstractmethod
    def is_finished(self, member, database):
        pass

class ExploitAndExplore(Controller):
    """ A general, modifiable implementation of PBTs exploitation and exploration method. """
    def __init__(self, exploit_factor = 0.2, explore_factors = (0.8, 1.2), frequency = 5, end_criteria = {'epoch': 20, 'score': 99.9}):
        assert isinstance(exploit_factor, float) and 0.0 <= exploit_factor <= 1.0, f"Exploit factor must be of type {float} between 0.0 and 1.0."
        assert isinstance(explore_factors, (float, list, tuple)), f"Explore factors must be of type {float}, {tuple} or {list}."
        assert isinstance(frequency, int) and frequency > 0, f"Frequency must be of type {int} as 1 or higher."
        assert isinstance(end_criteria, dict), f"End criteria must be of type {dict}."
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors
        self.frequency = frequency
        self.end_criteria = end_criteria

    def prepare(self, hyperparameters, logger):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        for hyperparameter_name, hyperparameter in hyperparameters:
            hyperparameter.sample_uniform()
            logger(f"{hyperparameter_name}: {hyperparameter.value()}")

    def evolve(self, member, database, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        population = database.get_latest()
        # sort members
        population.sort(key=lambda m: m.score, reverse=True)
        # set number of elitists
        n_elitists = math.floor(len(population) * self.exploit_factor)
        if n_elitists > 0 and all(c.id != member.id for c in population[:n_elitists]):
            # exploit weights and hyper-parameters if member is not elitist
            self.exploit(member, population[:n_elitists], logger)
        # explore new hyper-parameters
        self.explore(member, logger)

    def exploit(self, member, population, logger):
        """A fraction of the bottom performing members exploit the top performing members."""
        elitist = random.choice(population)
        logger(f"exploiting m{elitist.id}...")
        member.update(elitist)

    def explore(self, member, logger):
        """Perturb all parameters by the defined explore_factors."""
        logger("exploring...")
        for hyperparameter_name, hyperparameter in member.hyperparameters:
            old_value = hyperparameter.value()
            hyperparameter *= random.choice(self.explore_factors)
            new_value = hyperparameter.value()
            logger(f"{hyperparameter_name}: {old_value} --> {new_value}")

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