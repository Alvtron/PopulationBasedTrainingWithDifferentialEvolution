import math
import random
import copy
from abc import ABC, abstractmethod 

class EvolveEngine(ABC):
    @abstractmethod
    def prepare(self, hyper_parameters, logger = None):
        pass
    @abstractmethod
    def evolve(self, member, generation, population, function, logger):
        pass

class ExploitAndExplore(EvolveEngine):
    """ A general, modifiable implementation of PBTs exploitation and exploration method. """
    def __init__(self, N, exploit_factor = 0.2, explore_factors = (0.8, 1.2), random_walk=False):
        assert isinstance(exploit_factor, float) and 0.0 <= exploit_factor <= 1.0, f"Exploit factor must be of type {float} between 0.0 and 1.0."
        assert isinstance(explore_factors, (float, list, tuple)), f"Explore factors must be of type {float}, {tuple} or {list}."
        self.N = N
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors
        self.random_walk = random_walk

    def prepare(self, hyper_parameters, logger = None):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        logger(f"Preparing hyper-parameters...")
        for hyperparameter_name, hyperparameter in hyper_parameters:
            hyperparameter.sample_uniform()

    def evolve(self, member, generation, population, function, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        generation = generation()
        # check population size
        generation_size = len(generation)
        if generation_size != self.N:
            logger(f"The current generation available is not equal to the targeted generation size. ({generation_size} != {self.N})")
            return
        # set number of elitists
        n_elitists = math.floor(len(generation) * self.exploit_factor)
        if n_elitists > 0:
            # sort members from best to worst on score
            generation.sort(key=lambda m: m.score, reverse=True)
            if all(c.id != member.id for c in generation[:n_elitists]):
                # exploit weights and hyper-parameters if member is not elitist
                self.exploit(member, generation[:n_elitists], logger)
        # explore new hyper-parameters
        self.explore(member, logger)

    def exploit(self, member, generation, logger):
        """A fraction of the bottom performing members exploit the top performing members."""
        elitist = random.choice(generation)
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
    def __init__(self, N, F = 0.2, Cr = 0.8, constraint='clip'):
        if N < 3:
            raise ValueError("Population size 'N' must be at least 3 or higher.")
        self.N = N
        self.F = F
        self.Cr = Cr
        self.constraint = constraint

    def prepare(self, hyper_parameters, logger = None):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        logger(f"Preparing hyper-parameters...")
        for hyperparameter_name, hyperparameter in hyper_parameters:
            hyperparameter.set_constraint(self.constraint)
            hyperparameter.sample_uniform()

    def evolve(self, member, generation, population, function, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        generation = generation()
        # check population size
        generation_size = len(generation)
        if generation_size != self.N:
            logger(f"The current generation available is not equal to the targeted generation size. ({generation_size} != {self.N})")
            return
        hp_dimension_size = len(member.hyper_parameters)
        r0, r1, r2 = random.sample(range(0, generation_size), 3)
        j_rand = random.choice(range(0, hp_dimension_size))
        mutation = copy.deepcopy(member)
        for j in range(0, hp_dimension_size):
            if random.uniform(0.0, 1.0) <= self.Cr or j == j_rand:
                x_r0 = generation[r0].hyper_parameters[j]
                x_r1 = generation[r1].hyper_parameters[j]
                x_r2 = generation[r2].hyper_parameters[j]
                mutation.hyper_parameters[j].normalized = x_r0.normalized + (x_r1.normalized - x_r2.normalized) * self.F
            else:
                mutation.hyper_parameters[j] = member.hyper_parameters[j]
        # eval mutation
        mutation_score = function(mutation)
        if mutation_score >= member.score:
            logger(f"mutated member. (u {mutation_score:.4f} >= x {member.score:.4f})")
            member.update(mutation)
        else:
            logger(f"maintained member. (u {mutation_score:.4f} < x {member.score:.4f})")

class ParticleSwarm(EvolveEngine):
    """A general, modifiable implementation of Particle Swarm Optimization (PSO)"""
    def __init__(self, a = 0.2, b = (0.8, 1.2)):
        self.a = a
        self.b = b

    def prepare(self, hyper_parameters, logger = None):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        logger(f"Preparing hyper-parameters...")
        for hyperparameter_name, hyperparameter in hyper_parameters:
            hyperparameter.sample_uniform()

    def evolve(self, member, generation, population, function, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        random_p = random.uniform(0.0, 1.0)
        random_g = random.uniform(0.0, 1.0)
        # get members
        generation = generation()
        population = population()
        # set best member in current generation
        best_member_in_generation =  max(generation, key=lambda m: m.score)
        # set best member across generations
        best_overall_in_population =  max(population, key=lambda m: m.score)

    def update_velocity(self, particle, best_particle_in_generation, best_particle_across_generations, weight, velocity, acc_coeff_p, random_p, acc_coeff_g, random_g):
        return weight * velocity + acc_coeff_p * random_p * (best_particle_in_generation - particle) + acc_coeff_g * random_g * (best_particle_across_generations - particle)