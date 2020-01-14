import math
import random
from hyperparameters import Hyperparameters
from abc import abstractmethod
from member import MemberState

class EvolveEngine(object):
    def __init__(self, population_size):
        self.population_size = population_size

    @abstractmethod
    def prepare(self, hyper_parameters : Hyperparameters, logger = None):
        pass

    @abstractmethod
    def evolve(self, member : MemberState, population, function, logger):
        pass

class ExploitAndExplore(EvolveEngine):
    """ A general, modifiable implementation of PBTs exploitation and exploration method. """
    def __init__(self, population_size, exploit_factor = 0.2, explore_factors = (0.8, 1.2), random_walk=False):
        super().__init__(population_size)
        assert isinstance(exploit_factor, float) and 0.0 <= exploit_factor <= 1.0, f"Exploit factor must be of type {float} between 0.0 and 1.0."
        assert isinstance(explore_factors, (float, list, tuple)), f"Explore factors must be of type {float}, {tuple} or {list}."
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors
        self.random_walk = random_walk

    def prepare(self, hyper_parameters : Hyperparameters, logger = None):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        for hyper_parameter in hyper_parameters.parameters():
            hyper_parameter.sample_uniform()

    def evolve(self, member : MemberState, population, function, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        # check population size
        population = list(population)
        population_size = len(population)
        if population_size != self.population_size:
            logger(f"provided population is too small ({population_size} != {self.population_size}). Skipping.")
            return
        # set number of elitists
        n_elitists = math.floor(population_size * self.exploit_factor)
        if n_elitists > 0:
            # sort members from best to worst on score
            population.sort(key=lambda x: x.score(), reverse=not member.minimize)
            if all(m.id != member.id for m in population[:n_elitists]):
                # exploit weights and hyper-parameters if member is not elitist
                self.exploit(member, population[:n_elitists], logger)
        # explore new hyper-parameters
        self.explore(member, logger)

    def exploit(self, member : MemberState, population, logger):
        """A fraction of the bottom performing members exploit the top performing members."""
        elitist = random.choice(population)
        logger(f"exploiting m{elitist.id}...")
        member.update(elitist)

    def explore(self, member, logger):
        """Perturb all parameters by the defined explore_factors."""
        logger("exploring...")
        for _, hp in member.hyper_parameters:
            if not self.random_walk:
                perturb_factor = random.choice(self.explore_factors)
                hp *= perturb_factor
            else:
                min = 1.0 - self.explore_factors[0]
                max = 1.0 - self.explore_factors[1]
                walk_factor = random.uniform(min, max)
                hp += walk_factor

class DifferentialEvolution(EvolveEngine):
    """A general, modifiable implementation of Differential Evolution (DE)"""
    def __init__(self, population_size, F = 0.2, Cr = 0.8, constraint='clip'):
        if population_size < 3:
            raise ValueError("population size must be at least 3 or higher.")
        super().__init__(population_size)
        self.F = F
        self.Cr = Cr
        self.constraint = constraint

    def prepare(self, hyper_parameters : Hyperparameters, logger = None):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        for hyper_parameter in hyper_parameters.parameters():
            hyper_parameter.set_constraint(self.constraint)
            hyper_parameter.sample_uniform()

    def evolve(self, member : MemberState, population, function, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        # check population size
        population = list(population)
        population_size = len(population)
        if population_size != self.population_size:
            logger(f"provided population is too small ({population_size} != {self.population_size}). Skipping.")
            return
        hp_dimension_size = len(member.hyper_parameters)
        r0, r1, r2 = random.sample(range(0, population_size), 3)
        j_rand = random.choice(range(0, hp_dimension_size))
        mutation = member.copy()
        for j in range(0, hp_dimension_size):
            if random.uniform(0.0, 1.0) <= self.Cr or j == j_rand:
                x_r0 = population[r0].hyper_parameters[j]
                x_r1 = population[r1].hyper_parameters[j]
                x_r2 = population[r2].hyper_parameters[j]
                mutation.hyper_parameters[j].normalized = x_r0.normalized + (x_r1.normalized - x_r2.normalized) * self.F
            else:
                mutation.hyper_parameters[j] = member.hyper_parameters[j]
        # eval mutation
        mutation_score = function(mutation)
        if member < mutation_score :
            logger(f"mutate member (x {member.score():.4f} < u {mutation_score:.4f}).")
            member.update(mutation)
        else:
            logger(f"maintain member (x {member.score():.4f} > u {mutation_score:.4f}).")

class ParticleSwarm(EvolveEngine):
    """A general, modifiable implementation of Particle Swarm Optimization (PSO)"""
    def __init__(self, population_size, a = 0.2, b = (0.8, 1.2)):
        super().__init__(population_size)
        self.a = a
        self.b = b
        self.best_member = None

    def prepare(self, hyper_parameters : Hyperparameters, logger = None):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        for hyper_parameter in hyper_parameters.parameters():
            hyper_parameter.sample_uniform()

    def evolve(self, member : MemberState, population, function, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        population = list(population)
        population_size = len(population)
        if population_size != self.population_size:
            logger(f"provided population is too small ({population_size} != {self.population_size}). Skipping.")
            return
        random_p = random.uniform(0.0, 1.0)
        random_g = random.uniform(0.0, 1.0)
        # set best member in current population
        best_member = max(population, key=lambda m: m.score())
        if not self.best_member or best_member.score() > self.best_member.score():
            self.best_member = best_member

    def update_velocity(self, particle, best_particle_in_population, best_particle_across_populations, weight, velocity, acc_coeff_p, random_p, acc_coeff_g, random_g):
        return weight * velocity + acc_coeff_p * random_p * (best_particle_in_population - particle) + acc_coeff_g * random_g * (best_particle_across_populations - particle)