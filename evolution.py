import os
import math
import operator
import random
import copy
import torch
import time
from abc import ABC, abstractmethod 
from hyperparameters import Hyperparameter
from torch.utils.data import DataLoader
from database import Checkpoint, SharedDatabase
from utils import get_datetime_string

mp = torch.multiprocessing.get_context('spawn')

class EvolveEngine(ABC):
    def prepare(self, hyper_parameters, logger):
        pass

    def evolve(self, member, generation, population, function, logger):
        pass

class ExploitAndExplore(EvolveEngine):
    """ A general, modifiable implementation of PBTs exploitation and exploration method. """
    def __init__(self, exploit_factor = 0.2, explore_factors = (0.8, 1.2)):
        assert isinstance(exploit_factor, float) and 0.0 <= exploit_factor <= 1.0, f"Exploit factor must be of type {float} between 0.0 and 1.0."
        assert isinstance(explore_factors, (float, list, tuple)), f"Explore factors must be of type {float}, {tuple} or {list}."
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors

    def prepare(self, hyper_parameters, logger):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        logger(f"Preparing hyper-parameters...")
        for hyperparameter_name, hyperparameter in hyper_parameters:
            hyperparameter.sample_uniform()
            logger(f"{hyperparameter_name}: {hyperparameter.value()}")

    def evolve(self, member, generation, population, function, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        generation = generation()
        # set number of elitists
        n_elitists = math.floor(len(generation) * self.exploit_factor)
        if n_elitists > 0:
            # sort members from best to worst on score
            generation.sort(key=lambda m: m.eval_score, reverse=True)
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
            #perturb_factor = random.choice(self.explore_factors)
            #hp *= perturb_factor
            perturb_factor = random.uniform(-0.2, 0.2)
            hp += perturb_factor

class DifferentialEvolution(EvolveEngine):
    """A general, modifiable implementation of Differential Evolution (DE)"""
    def __init__(self, N, F = 0.2, Cr = 0.8):
        self.N = N
        self.F = F
        self.Cr = Cr

    def prepare(self, hyper_parameters, logger):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        logger(f"Preparing hyper-parameters...")
        for hyperparameter_name, hyperparameter in hyper_parameters:
            hyperparameter.sample_uniform()
            logger(f"{hyperparameter_name}: {hyperparameter.value()}")

    def evolve(self, member, generation, population, function, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        generation = generation()
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
                mutation.hyper_parameters[j] = x_r0 + (x_r1 - x_r2) * self.F
            else:
                mutation.hyper_parameters[j] = member.hyper_parameters[j]
        member_score = function(member)
        mutation_score = function(mutation)
        if mutation_score >= member_score:
            logger(f"Mutated member. (u {mutation_score:.4f} >= x {member_score:.4f})")
            member.update(mutation)
        else:
            logger(f"Maintained member. (u {mutation_score:.4f} < x {member_score:.4f})")

class ParticleSwarm(EvolveEngine):
    """A general, modifiable implementation of Particle Swarm Optimization (PSO)"""
    def __init__(self, a = 0.2, b = (0.8, 1.2)):
        self.a = a
        self.b = b

    def prepare(self, hyper_parameters, logger):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        logger(f"Preparing hyper-parameters...")
        for hyperparameter_name, hyperparameter in hyper_parameters:
            hyperparameter.sample_uniform()
            logger(f"{hyperparameter_name}: {hyperparameter.value()}")

    def evolve(self, member, generation, population, function, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        random_p = random.uniform(0.0, 1.0)
        random_g = random.uniform(0.0, 1.0)
        # get members
        generation = generation()
        population = population()
        # set best member in current generation
        best_member_in_generation =  max(generation, key=lambda m: m.eval_score)
        # set best member across generations
        best_overall_in_population =  max(population, key=lambda m: m.eval_score)

    def update_velocity(self, particle, best_particle_in_generation, best_particle_across_generations, weight, velocity, acc_coeff_p, random_p, acc_coeff_g, random_g):
        return weight * velocity + acc_coeff_p * random_p * (best_particle_in_generation - particle) + acc_coeff_g * random_g * (best_particle_across_generations - particle)