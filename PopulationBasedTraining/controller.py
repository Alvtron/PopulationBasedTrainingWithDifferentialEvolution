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

class DifferentialEvolution(Controller):
    """A general, modifiable implementation of Differential Evolution (DE)"""
    def __init__(self, N, F = 0.2, Cr = 0.8, frequency = 5, max_generations = 20):
        self.N = N
        self.F = F
        self.Cr = Cr
        self.frequency = frequency
        self.max_generations = max_generations

    def prepare(self, hyperparameters, logger):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        for param_name, param in hyperparameters:
            param.sample_uniform()
            logger(f"{param_name}: {param.value()}")

    def evolve(self, member, database, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        population = database.get_lates()
        population_size = len(population)
        hp_dimension_size = len(member.hyperparameters)
        r0, r1, r2 = random.sample(range(0, population_size), 3)
        j_rand = random.sample(range(0, hp_dimension_size))
        mutation_vector = member.hyperparameters
        for j in range(0, hp_dimension_size):
            if random.uniform(0.0, 1.0) <= self.Cr or j == j_rand:
                mutation_vector[j] = population[r0][j] + self.F * (population[r1][j] - population[r2][j])
            else:
                mutation_vector[j] = member.hyperparameters
        mutation_score = eval(mutation)
        if mutation_score >= member.score:
            member.hyperparameters = mutation_vector
        else:
            member.hyperparameters = member.hyperparameters


    def update_velocity(self, particle, best_particle_in_generation, best_particle_across_generations, weight, velocity, acc_coeff_p, random_p, acc_coeff_g, random_g):
        return weight * velocity + acc_coeff_p * random_p * (best_particle_in_generation - particle) + acc_coeff_g * random_g * (best_particle_across_generations - particle)
    
    def is_ready(self, member, database):
        """True every n-th epoch."""
        return member.epoch % self.frequency == 0

    def is_finished(self, member, database):
        """ True if a given number of generations have passed"""
        return member.epoch > self.max_generations

class ParticleSwarm(Controller):
    """A general, modifiable implementation of Particle Swarm Optimization (PSO)"""
    def __init__(self, a = 0.2, b = (0.8, 1.2), frequency = 5, max_generations = 20):
        self.a = a
        self.b = b
        self.frequency = frequency
        self.max_generations = max_generations

    def prepare(self, hyperparameters, logger):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        for hyperparameter_name, hyperparameter in hyperparameters:
            hyperparameter.sample_uniform()
            logger(f"{hyperparameter_name}: {hyperparameter.value()}")

    def evolve(self, member, database, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        random_p = random.uniform(0.0, 1.0)
        random_g = random.uniform(0.0, 1.0)
        # get members
        latest_members = database.get_latest()
        all_members = database.to_list()
        # set best member in current generation
        best_member_in_generation =  max(latest_members, key=operator.attrgetter('score'))
        # set best member across generations
        best_overall_member =  max(all_members, key=operator.attrgetter('score'))

    def update_velocity(self, particle, best_particle_in_generation, best_particle_across_generations, weight, velocity, acc_coeff_p, random_p, acc_coeff_g, random_g):
        return weight * velocity + acc_coeff_p * random_p * (best_particle_in_generation - particle) + acc_coeff_g * random_g * (best_particle_across_generations - particle)
    
    def is_ready(self, member, database):
        """True every n-th epoch."""
        return member.epoch % self.frequency == 0

    def is_finished(self, member, database):
        """ True if a given number of generations have passed"""
        return member.epoch > self.max_generations