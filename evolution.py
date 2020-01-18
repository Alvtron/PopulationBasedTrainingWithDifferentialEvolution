import math
import random
import copy
import warnings
import collections
from hyperparameters import Hyperparameters
from abc import abstractmethod
from typing import Tuple, List, Dict
from member import MemberState
from de.mutation import de_rand_1, de_current_to_best_1
from de.constraint import halving
from utils.constraint import clip
from utils.distribution import randn, randc

def random_from_dict(members : Dict[object, MemberState], k : int = 1, exclude_ids=None):
    if not isinstance(exclude_ids, str) and not isinstance(exclude_ids, collections.Sequence):
        exclude_ids = [exclude_ids]
    filtered = members if not exclude_ids else [member for id, member in members.items() if member.id not in exclude_ids]
    return random.sample(filtered, k) if k and k > 1 else random.choice(filtered)

def random_from_list(members : List[MemberState], k : int = 1, exclude_ids=None):
    if not isinstance(exclude_ids, str) and not isinstance(exclude_ids, collections.Sequence):
        exclude_ids = [exclude_ids]
    filtered = members if not exclude_ids else [member for member in members if member.id not in exclude_ids]
    return random.sample(filtered, k) if k and k > 1 else random.choice(filtered)

class EvolveEngine(object):
    """
    Base class for all evolvers.
    """
    def __init__(self, population_size):
        self.population_size = population_size
        self.population = dict()

    def on_initialization(self, population : Dict[object, MemberState], logger):
        """Called before optimization begins."""
        pass

    @abstractmethod
    def on_generation_start(self, population : Dict[object, MemberState], logger):
        """Called before each generation."""
        pass

    @abstractmethod
    def on_evolve(self, index : int, population : Dict[object, MemberState], function, logger):
        """Called for each member in generation."""
        pass

    @abstractmethod
    def on_evaluation(self, member : MemberState, trial : MemberState, eval_function, logger):
        """Called for each member in generation."""
        pass

    @abstractmethod
    def on_generation_end(self, population : Dict[object, MemberState], logger):
        """Called at the end of each generation."""
        pass

class ExploitAndExplore(EvolveEngine):
    """
    A general, modifiable implementation of PBTs exploitation and exploration method.
    """
    def __init__(self, population_size, exploit_factor = 0.2, explore_factors = (0.8, 1.2), random_walk=False, constraint='clip'):
        super().__init__(population_size)
        assert isinstance(exploit_factor, float) and 0.0 <= exploit_factor <= 1.0, f"Exploit factor must be of type {float} between 0.0 and 1.0."
        assert isinstance(explore_factors, (float, list, tuple)), f"Explore factors must be of type {float}, {tuple} or {list}."
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors
        self.random_walk = random_walk
        self.constraint = constraint

    def on_initialization(self, population : Dict[object, MemberState], logger):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        for member in population.values():
            for hyper_parameter in member.hyper_parameters.parameters():
                hyper_parameter.set_constraint(self.constraint)
                hyper_parameter.sample_uniform()

    def on_generation_start(self, logger):
        pass

    def on_evolve(self, member : MemberState, logger):
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        # exploit population
        exploiter = self.exploit(member, self.population, logger)
        logger("exploring...")
        explorer = self.explore(exploiter, logger)
        return explorer

    def on_evaluation(self, member : MemberState, trial : MemberState, eval_function, logger):
        """Simply returns the trial. No evaluation conducted."""
        return trial

    def exploit(self, member : MemberState, population, logger):
        """A fraction of the bottom performing members exploit the top performing members."""
        exploiter = member.copy()
        n_elitists = math.floor(len(population) * self.exploit_factor)
        if n_elitists == 0:
            return exploiter
        elitists = sorted(population.values(), reverse=True)[:n_elitists]
        # exploit if member is not elitist
        if all(m.id != member.id for m in elitists):
            elitist = random.choice(elitists)
            logger(f"exploiting member {elitist.id}...")
            exploiter.update(elitist)
        else:
            logger(f"exploitation not needed.")
        return exploiter

    def explore(self, member, logger):
        """Perturb all parameters by the defined explore_factors."""
        explorer = member.copy()
        for _, hp in explorer.hyper_parameters:
            if not self.random_walk:
                perturb_factor = random.choice(self.explore_factors)
                hp *= perturb_factor
            else:
                min = 1.0 - self.explore_factors[0]
                max = 1.0 - self.explore_factors[1]
                walk_factor = random.uniform(min, max)
                hp += walk_factor
        return explorer
    
    def on_generation_end(self, logger):
        pass

class DifferentialEvolution(EvolveEngine):
    """
    A general, modifiable implementation of Differential Evolution (DE)
    """
    def __init__(self, population_size, F = 0.2, Cr = 0.8, constraint='clip'):
        if population_size < 3:
            raise ValueError("population size must be at least 3 or higher.")
        super().__init__(population_size)
        self.F = F
        self.Cr = Cr
        self.constraint = constraint

    def on_initialization(self, population : Dict[object, MemberState], logger):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        for member in population.values():
            for hyper_parameter in member.hyper_parameters.parameters():
                hyper_parameter.set_constraint(self.constraint)
                hyper_parameter.sample_uniform()

    def on_generation_start(self, logger):
        pass

    def on_evolve(self, member : MemberState, logger):
        """
        Perform crossover, mutation and selection according to the initial 'DE/rand/1/bin'
        implementation of differential evolution.
        """
        # check population size
        hp_dimension_size = len(member.hyper_parameters)
        x_r0, x_r1, x_r2 = random_from_dict(self.population, k=3, exclude_ids=member.id)
        j_rand = random.randrange(0, hp_dimension_size)
        for j in range(hp_dimension_size):
            base = member.hyper_parameters[j].normalized
            if random.uniform(0.0, 1.0) <= self.Cr or j == j_rand:
                mutant = de_rand_1(
                    F = self.F,
                    x_r0 = x_r0.hyper_parameters[j].normalized,
                    x_r1 = x_r1.hyper_parameters[j].normalized,
                    x_r2 = x_r2.hyper_parameters[j].normalized)
                member.hyper_parameters[j].normalized = mutant
            else:
                member.hyper_parameters[j].normalized = base
        return member

    def on_evaluation(self, member : MemberState, trial : MemberState, eval_function, logger):
        """Evaluates trial, compares it to the base and returns the best performer."""
        trial = eval_function(trial)
        if member <= trial :
            logger(f"mutate member (x {member.score():.4f} <= u {trial.score():.4f}).")
            return trial
        else:
            logger(f"maintain member (x {member.score():.4f} > u {trial.score():.4f}).")
            return member

    def on_generation_end(self, logger):
        pass

def mean_wl(S, weights):
    def weight(weights, k):
        return weights[k] / sum(weights)
    A = sum(weight(weights, k) * s**2 for k, s in enumerate(S))
    B = sum(weight(weights, k) * s for k, s in enumerate(S))
    return A / B

class HistoricalMemory(object):
    def __init__(self, size):
        self.size = size
        self.m_cr = [0.5] * size
        self.m_f = [0.5] * size
        self.s_cr = list()
        self.s_f = list()
        self.weights = list()
        self.k = 0
    
    def record(self, cr_i, f_i, delta_score):
        self.s_cr.append(cr_i)
        self.s_f.append(f_i)
        self.weights.append(delta_score)

    def update(self):
        if self.s_cr and self.s_f:
            if self.m_cr[self.k] == None or max(self.s_cr) == 0:
                self.m_cr[self.k] = None
            else:
                self.m_cr[self.k] = mean_wl(self.s_cr, self.weights)
            self.m_f[self.k] = mean_wl(self.s_f, self.weights)
            self.k = 0 if self.k >= self.size - 1 else self.k + 1

class ExternalArchive(object):
    def __init__(self, size):
        self.size = size
        self.parents = list()

    def add_parent(self, parent):
        if len(self.parents) == self.size:
            random_index = random.randrange(self.size)
            self.parents.pop(random_index)
        self.parents.append(parent)

class LSHADE(EvolveEngine):
    """
    A general, modifiable implementation of Success-History based Adaptive Differential Evolution (SHADE)
    with linear population size reduction.

    References:
        SHADE: https://ieeexplore.ieee.org/document/6557555
        L-SHADE: https://ieeexplore.ieee.org/document/6900380

    Parameters:
        population_size: The population size, also denoted N_INIT {15, 16, ..., 25}.
        r_arc: adjusts archive size with round(N_INIT * rarc) {1.0, 1.1, ..., 3.0}.
        MAX_NFE: the maximum number of fitness evaluations (N * (end_steps / step_size))
        p: control parameter for DE/current-to-pbest/1/. Small p, more greedily {0.05, 0.06, ..., 0.15}.
        memory_size: historical memory size (H) {2, 3, ..., 10}.
    """
    def __init__(self, population_size, MAX_NFE, r_arc = 2.0, p=0.1, memory_size = 5):
        if population_size < 4:
            raise ValueError("population size must be at least 4 or higher.")
        if round(population_size * p) < 1:
            warnings.warn(f"p-parameter too low for the provided population size. It must be atleast {1.0 / population_size} for population size of {population_size}. This will be resolved by always choosing the top one performer in the population as pbest.")
        super().__init__(population_size)
        self.N_INIT = population_size
        self.N_MIN = 4
        self.NFE = 0
        self.MAX_NFE = MAX_NFE
        self.archive = ExternalArchive(round(self.N_INIT * r_arc))
        self.memory = HistoricalMemory(memory_size)
        self.p = p
        self.mutation = de_current_to_best_1
        self.constraint = halving
        self.CR_i = None
        self.F_i = None
        
    def on_initialization(self, population : Dict[object, MemberState], logger):
        """For every hyperparameter, sample a new random, uniform sample within the constrained search space."""
        for member in population.values():
            for hyper_parameter in member.hyper_parameters.parameters():
                hyper_parameter.sample_uniform()

    def on_generation_start(self, logger):
        self.archive.s_cr = list()
        self.archive.s_f = list()

    def on_evolve(self, member : MemberState, logger):
        """
        Perform crossover, mutation and selection according to the initial 'DE/rand/1/bin'
        implementation of differential evolution.
        """
        # check population size
        hp_dimension_size = len(member.hyper_parameters)
        # control parameter assignment
        self.CR_i, self.F_i = self.get_control_parameters()
        # random unique members
        if self.archive.parents:
            x_r1 = random_from_dict(self.population, exclude_ids=member.id)
            x_r2 = random_from_list(self.archive.parents)
        else:
            x_r1, x_r2 = random_from_dict(self.population, k=2, exclude_ids=member.id)
        # random best member
        x_pbest = self.pbest_member()
        # random parameter dimension
        j_rand = random.randrange(0, hp_dimension_size)
        for j in range(hp_dimension_size):
            base = member.hyper_parameters[j].normalized
            if random.uniform(0.0, 1.0) <= self.CR_i or j == j_rand:
                mutant = self.mutation(
                    F = self.F_i,
                    x_base = base,
                    x_best = x_pbest.hyper_parameters[j].normalized,
                    x_r1 = x_r1.hyper_parameters[j].normalized,
                    x_r2 = x_r2.hyper_parameters[j].normalized)
                mutant = self.constraint(
                    base = base,
                    mutant = mutant,
                    lower_bounds = 0.0,
                    upper_bounds = 1.0
                )
                member.hyper_parameters[j].normalized = mutant
            else:
                member.hyper_parameters[j].normalized = base
        return member
    
    def on_evaluation(self, member : MemberState, trial : MemberState, eval_function, logger):
        """Evaluates trial, compares it to the base and returns the best performer."""
        trial = eval_function(trial)
        self.NFE += 1 # increment the number of fitness evaluations
        if member < trial:
            logger(f"mutation is better. Add parent member to archive.")
            self.archive.add_parent(member)
            self.memory.record(self.CR_i, self.F_i, abs(trial.score() - member.score()))
        if member <= trial:
            logger(f"mutate member (x {member.score():.4f} < u {trial.score():.4f}).")
            return trial
        else:
            logger(f"maintain member (x {member.score():.4f} > u {trial.score():.4f}).")
            return member

    def on_generation_end(self, logger):
        self.memory.update()
        self.adjust_population_size(logger)

    def adjust_population_size(self, logger):
        new_size = round(((self.N_MIN - self.N_INIT) / self.MAX_NFE) * self.NFE + self.N_INIT)
        sorted_members = sorted(self.population.values(), reverse=False)
        # if new size is smaller than the current size,
        # delete the delta worst performing members
        if new_size < self.population_size:
            logger(f"adjusting population size {self.population_size} --> {new_size}")
            for member in sorted_members[:self.population_size - new_size]:
                logger(f"member {member.id} is marked for removal")
                del self.population[member.id]
            self.population_size = new_size

    def get_control_parameters(self):
        r1 = random.randrange(0, self.memory.size)
        MF_i = self.memory.m_f[r1]
        MCR_i = self.memory.m_cr[r1]
        # constrain MCR_i
        CR_i = MCR_i
        if MCR_i == None:
            CR_i = 0.0
        CR_i = clip(randn(MCR_i, 0.1), 0.0, 1.0)
        # constrain MF_i
        F_i = randc(MF_i, 0.1)
        if F_i > 1.0:
            F_i = 1.0
        while F_i <= 0.0 or F_i > 1.0:
            F_i = randc(MF_i, 0.1)
        return CR_i, F_i
        
    def pbest_member(self):
        """Sample a random top member from the popualtion."""
        sorted_members = sorted(self.population.values(), reverse=True)
        n_elitists = round(len(self.population) * self.p)
        n_elitists = max(n_elitists, 1) # correction for too small p-values
        elitists = sorted_members[:n_elitists]
        return random.choice(elitists)
