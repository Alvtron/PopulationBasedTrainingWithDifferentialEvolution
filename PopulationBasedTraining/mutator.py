import abc
import math
import operator
import random
from utils import get_datetime_string
from database import Checkpoint, Database
from hyperparameters import Hyperparameter

class Mutator(abc.ABC):
    @abc.abstractmethod
    def apply_mutation(self, checkpoint, database):
        pass

    @abc.abstractmethod
    def is_ready(self, checkpoint, database):
        pass

    @abc.abstractmethod
    def is_finished(self, checkpoint, database):
        pass

class ExploitAndExplore(Mutator):
    """ A general, modifiable implementation of PBTs exploitation and exploration method. """
    def __init__(self, exploit_factor = 0.4, explore_factors = (1.2, 0.8), frequency = 5, end_criteria = {'epoch': 20, 'score': 99.9}):
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors
        self.frequency = frequency
        self.end_criteria = end_criteria
    
    def apply_mutation(self, checkpoint, database):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        super().apply_mutation(checkpoint, database)
        population = database.get_latest()
        # sort checkpoints
        population.sort(key=lambda m: m.score, reverse=True)
        # set number of elitists
        n_elitists = math.floor(len(population) * self.exploit_factor)
        if n_elitists > 0 and all(c.id != checkpoint.id for c in population[:n_elitists]):
            # exploit
            self.exploit(checkpoint, population[:n_elitists])
        # explore
        self.explore(checkpoint)

    def is_ready(self, checkpoint, database):
        """True every n-th epoch."""
        return checkpoint.epoch % self.frequency == 0

    def is_finished(self, checkpoint, database):
        """ True if a given number of epochs have passed or if the score reaches above 99%. """
        if 'epoch' in self.end_criteria and checkpoint.epoch > self.end_criteria['epoch']:
            # the number of epochs is above the given treshold
            return True
        all_checkpoints = database.to_list()
        if not all_checkpoints:
            # database is currently empty
            return False
        best_score = max(c.score for c in all_checkpoints)
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
            for parameter_group in member.optimizer_state['param_groups']:
                # create a random perturbation factor with the given perturb factors
                perturb_factor = random.choice(self.explore_factors)
                hyperparameter.perturb(perturb_factor)
                parameter_group[hyperparameter_name] = hyperparameter.value
        # exploring batch_size
        if member.hyperparameters['batch_size']:
            # create a random perturbation factor with the given perturb factors
            perturb_factor = random.choice(self.explore_factors)
            member.hyperparameters['batch_size'].perturb(perturb_factor)
            member.batch_size = member.hyperparameters['batch_size'].value