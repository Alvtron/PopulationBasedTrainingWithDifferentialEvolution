import math
import random
import copy
import warnings
import collections
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Iterable, Sequence, Callable, Generator

from .member import MemberState, Generation, Population
from .hyperparameters import Hyperparameters
from .de.mutation import de_rand_1, de_current_to_best_1
from .de.constraint import halving
from .utils.constraint import clip
from .utils.distribution import randn, randc
from .utils.iterable import grid, random_from_list, average

class EvolveEngine(ABC):
    """
    Base class for all evolvers.
    """
    
    def on_spawn(self, member : MemberState, logger : Callable[[str], None]) -> None:
        """Called for each new member."""
        pass

    def on_generation_start(self, generation : Generation, logger : Callable[[str], None]) -> None:
        """Called before each generation."""
        pass

    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> Generator[Tuple[MemberState, ...], None, None]:
        """Called for each member in generation. Returns one candidate or multiple candidates."""
        pass

    def on_evaluate(self, candidates : Tuple[MemberState, ...], logger : Callable[[str], None]) -> MemberState:
        """Returns the determined 'best' member between the candidates."""
        pass

    def on_generation_end(self, generation : Generation, logger : Callable[[str], None]) -> None:
        """Called at the end of each generation."""
        pass

class RandomSearch(EvolveEngine):
    def __init__(self):
        super().__init__()

    def on_spawn(self, member : MemberState, logger : Callable[[str], None]) -> None:
        """Called for each new member."""
        [hp.sample_uniform() for hp in member.parameters]

    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> MemberState:
        """Simply returns the member. No mutation conducted."""
        for member in generation:
            yield member.copy()

    def on_evaluate(self, candidate : MemberState, logger : Callable[[str], None]) -> MemberState:
        """Simply returns the candidate. No evaluation conducted."""
        return candidate

class RandomWalk(EvolveEngine):
    def __init__(self, explore_factor = 0.2) -> None:
        super().__init__()
        self.explore_factor = explore_factor

    def on_spawn(self, member : MemberState, logger : Callable[[str], None]) -> None:
        """Called for each new member."""
        [hp.sample_uniform() for hp in member.parameters]

    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> MemberState:
        """ Explore search space with random walk. """
        for member in generation:
            logger(f"exploring member {member.id}...")
            explorer = member.copy()
            for index, _ in enumerate(explorer.parameters):
                perturb_factor = random.uniform(-self.explore_factor, self.explore_factor)
                explorer[index] = explorer[index] * perturb_factor
            yield explorer

    def on_evaluate(self, candidate : MemberState, logger : Callable[[str], None]) -> MemberState:
        """Simply returns the candidate. No evaluation conducted."""
        return candidate

class ExploitAndExplore(EvolveEngine):
    """
    A general, modifiable implementation of PBTs exploitation and exploration method.
    """
    def __init__(self, exploit_factor : float = 0.2, explore_factors : Tuple[float, ...] = (0.8, 1.2)) -> None:
        super().__init__()
        assert isinstance(exploit_factor, float) and 0.0 <= exploit_factor <= 1.0, f"Exploit factor must be of type {float} between 0.0 and 1.0."
        assert isinstance(explore_factors, (float, list, tuple)), f"Explore factors must be of type {float}, {tuple} or {list}."
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors

    def on_spawn(self, member : MemberState, logger : Callable[[str], None]) -> None:
        """Called for each new member."""
        [hp.sample_uniform() for hp in member.parameters]

    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> MemberState:
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        for member in generation:
            candidate, is_exploited = self.exploit(member, generation, logger)
            if is_exploited:
                candidate = self.explore(candidate, logger)
            yield candidate

    def on_evaluate(self, candidate : MemberState, logger : Callable[[str], None]) -> MemberState:
        """Simply returns the candidate. No evaluation conducted."""
        return candidate

    def exploit(self, member : MemberState, generation : Generation, logger : Callable[[str], None]) -> Tuple[MemberState, bool]:
        """A fraction of the bottom performing members exploit the top performing members."""
        exploiter = member.copy()
        n_elitists = max(1, round(len(generation) * self.exploit_factor))
        sorted_members = sorted(generation, reverse=True)
        elitists = sorted_members[:n_elitists]
        # exploit if member is not elitist
        if member not in elitists:
            elitist = random.choice(elitists)
            logger(f"exploiting member {elitist.id}...")
            exploiter.copy_parameters(elitist)
            exploiter.copy_state(elitist)
            return exploiter, True
        else:
            return exploiter, False

    def explore(self, member : MemberState, logger : Callable[[str], None]) -> MemberState:
        """Perturb all parameters by the defined explore_factors."""
        logger(f"exploring member {member.id}...")
        explorer = member.copy()
        for index, _ in enumerate(explorer.parameters):
            perturb_factor = random.choice(self.explore_factors)
            explorer[index] = explorer[index] * perturb_factor
        return explorer

class BlindDifferentialEvolution(ExploitAndExplore):
    """
    A general, modifiable implementation of PBTs exploitation and exploration method.
    """
    def __init__(self, exploit_factor : float = 0.2, F : float = 0.2, Cr : float = 0.8) -> None:
        super().__init__()
        self.exploit_factor = exploit_factor
        self.F = F
        self.Cr = Cr

    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> Tuple[MemberState, MemberState]:
        """
        Exploit best members in generation.\n
        Perform crossover, mutation and selection according to the initial 'DE/rand/1/bin' implementation of differential evolution.
        """
        for member in generation:
            candidate, has_exploited = self.exploit(member, generation, logger)
            if has_exploited:
                candidate = self.explore(candidate, generation, logger)
            yield candidate

    def on_evaluate(self, candidates : Tuple[MemberState, MemberState], logger : Callable[[str], None]) -> MemberState:
        """Evaluates candidate, compares it to the base and returns the best performer."""
        return candidates

    def explore(self, member : MemberState, generation : Generation, logger : Callable[[str], None]) -> MemberState:
        if len(generation) < 3:
            raise ValueError("generation size must be at least 3 or higher.")
        candidate = member.copy()
        hp_dimension_size = len(member.parameters)
        x_r0, x_r1, x_r2 = random_from_list(generation, k=3, exclude=(member,))
        j_rand = random.randrange(0, hp_dimension_size)
        for j in range(hp_dimension_size):
            if random.uniform(0.0, 1.0) <= self.Cr or j == j_rand:
                candidate[j] = de_rand_1(F = self.F, x_r0 = x_r0[j], x_r1 = x_r1[j], x_r2 = x_r2[j])
            else:
                candidate[j] = member[j]
        return candidate

class DifferentialEvolution(EvolveEngine):
    """
    A general, modifiable implementation of Differential Evolution (DE)
    """
    def __init__(self, F : float = 0.2, Cr : float = 0.8) -> None:
        super().__init__()
        self.F = F
        self.Cr = Cr

    def on_spawn(self, member : MemberState, logger : Callable[[str], None]):
        """Called for each new member."""
        [hp.sample_uniform() for hp in member.parameters]

    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> Tuple[MemberState, MemberState]:
        """
        Perform crossover, mutation and selection according to the initial 'DE/rand/1/bin'
        implementation of differential evolution.
        """
        if len(generation) < 3:
            raise ValueError("generation size must be at least 3 or higher.")
        for member in generation:
            candidate = member.copy()
            hp_dimension_size = len(member.parameters)
            x_r0, x_r1, x_r2 = random_from_list(generation, k=3, exclude=(member,))
            j_rand = random.randrange(0, hp_dimension_size)
            for j in range(hp_dimension_size):
                if random.uniform(0.0, 1.0) <= self.Cr or j == j_rand:
                    mutant = de_rand_1(F = self.F, x_r0 = x_r0[j], x_r1 = x_r1[j], x_r2 = x_r2[j])
                    constrained = clip(mutant, 0.0, 1.0)
                    candidate[j] = constrained 
                else:
                    candidate[j] = member[j]
            yield member.copy(), candidate

    def on_evaluate(self, candidates : Tuple[MemberState, MemberState], logger : Callable[[str], None]) -> MemberState:
        """Evaluates candidate, compares it to the base and returns the best performer."""
        member, candidate = candidates
        if member <= candidate :
            logger(f"mutate member {member.id} (x {member.score():.4f} <= u {candidate.score():.4f}).")
            return candidate
        else:
            logger(f"maintain member {member.id} (x {member.score():.4f} > u {candidate.score():.4f}).")
            return member

def mean_wl(S, weights : Sequence[float]) -> float:
    """
    The weighted Lehmer mean of a tuple x of positive real numbers,
    with respect to a tuple w of positive weights.
    """
    def weight(weights, k):
        return weights[k] / sum(weights)
    A = sum(weight(weights, k) * s**2 for k, s in enumerate(S))
    B = sum(weight(weights, k) * s for k, s in enumerate(S))
    return A / B

class HistoricalMemory(object):
    def __init__(self, size : int, default : float = 0.5) -> None:
        self.size = size
        self.m_cr = [default] * size
        self.m_f = [default] * size
        self.s_cr = list()
        self.s_f = list()
        self.weights = list()
        self.k = 0

    def record(self, cr_i : float, f_i : float, delta_score : float) -> None:
        self.s_cr.append(cr_i)
        self.s_f.append(f_i)
        self.weights.append(delta_score)
    
    def reset(self) -> None:
        self.s_cr = list()
        self.s_f = list()
        self.weights = list()

    def update(self) -> None:
        if not self.s_cr or not self.s_f or not self.weights:
            return
        if self.m_cr[self.k] == None or max(self.s_cr) == 0.0:
            self.m_cr[self.k] = None
        else:
            self.m_cr[self.k] = mean_wl(self.s_cr, self.weights)
        self.m_f[self.k] = mean_wl(self.s_f, self.weights)
        self.k = 0 if self.k >= self.size - 1 else self.k + 1

class ExternalArchive(list):
    def __init__(self, size : int) -> None:
        list.__init__(self)
        self.size = size

    def append(self, parent : MemberState) -> None:
        if len(self) == self.size:
            random_index = random.randrange(self.size)
            del self[random_index]
        super().append(parent)

    def insert(self, index, parent) -> None:
        raise NotImplementedError()
    
    def extend(self, parents) -> None:
        raise NotImplementedError()

class SHADE(EvolveEngine):
    """
    A general, modifiable implementation of Success-History based Adaptive Differential Evolution (SHADE).

    References:
        SHADE: https://ieeexplore.ieee.org/document/6557555

    Parameters:
        N_INIT: The number of members in the population {15, 16, ..., 25}.
        r_arc: adjusts archive size with round(N_INIT * rarc) {1.0, 1.1, ..., 3.0}.
        p: control parameter for DE/current-to-pbest/1/. Small p, more greedily {0.05, 0.06, ..., 0.15}.
        memory_size: historical memory size (H) {2, 3, ..., 10}.
    """
    def __init__(self, N_INIT : int, r_arc : float = 2.0, p : float = 0.1, memory_size : int = 5, f_min : float = 0.0, f_max : float = 1.0) -> None:
        if N_INIT < 4:
            raise ValueError("population size must be at least 4 or higher.")
        if round(N_INIT * p) < 1:
            warnings.warn(f"p-parameter too low for the provided population size. It must be atleast {1.0 / N_INIT} for population size of {N_INIT}. This will be resolved by always choosing the top one performer in the population as pbest.")
        super().__init__()
        if f_min < 0:
            raise ValueError("f_min cannot be negative.")
        if f_max <= 0:
            raise ValueError("f_max cannot be negative.")
        if f_max < f_min:
            raise ValueError("f_max cannot be less than f_min.")
        self.F_MIN = f_min
        self.F_MAX = f_max
        self.N_INIT = N_INIT
        self.r_arc = r_arc
        self.archive = ExternalArchive(size=round(self.N_INIT * r_arc))
        self.memory = HistoricalMemory(size=memory_size, default=(self.F_MAX - self.F_MIN) / 2.0)
        self.p = p
        self.CR = dict()
        self.F = dict()
        
    def on_spawn(self, member : MemberState, logger : Callable[[str], None]) -> None:
        """Called for each new member."""
        [hp.sample_uniform() for hp in member.parameters]

    def on_generation_start(self, generation : Generation, logger : Callable[[str], None]) -> None:
        self.memory.reset()
        self.CR = dict()
        self.F = dict()

    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> Tuple[MemberState, MemberState]:
        """
        Perform crossover, mutation and selection according to the initial 'DE/current-to-pbest/1/bin'
        implementation of differential evolution, with adapted CR and F parameters.
        """
        if len(generation) < 4:
            raise ValueError("generation size must be at least 4 or higher.")
        for index, member in generation.entries():
            # control parameter assignment
            self.CR[index], self.F[index] = self.get_control_parameters()
            # select random unique members from the union of the generation and archive
            x_r1, x_r2 = random_from_list(self.archive + list(generation), k=2, exclude=(member,))
            # select random best member
            x_pbest = self.pbest_member(generation)
            # hyper-parameter dimension size
            dimension_size = len(member.parameters)
            # choose random parameter dimension
            j_rand = random.randrange(0, dimension_size)
            # make a copy of the member
            candidate = member.copy()
            for j in range(dimension_size):
                if random.uniform(0.0, 1.0) <= self.CR[index] or j == j_rand:
                    mutant = de_current_to_best_1(F = self.F[index], x_base = member[j],
                        x_best = x_pbest[j], x_r1 = x_r1[j], x_r2 = x_r2[j])
                    constrained = halving(base = member[j], mutant = mutant,
                        lower_bounds = 0.0, upper_bounds = 1.0)
                    candidate[j] = constrained
                else:
                    candidate[j] = member[j]
            yield member.copy(), candidate

    def on_evaluate(self, candidates : Tuple[MemberState, MemberState], logger : Callable[[str], None]) -> MemberState:
        """Evaluates candidate, compares it to the original member and returns the best performer."""
        member, candidate = candidates
        if member <= candidate:
            logger(f"mutate member {member.id} (x {member.score():.4f} < u {candidate.score():.4f}).", member)
            if member < candidate:
                logger(f"adding parent member {member.id} to archive.", member)
                self.archive.append(member.copy())
                self.memory.record(self.CR[member.id], self.F[member.id], abs(candidate.score() - member.score()))
            return candidate
        else:
            logger(f"maintain member {member.id} (x {member.score():.4f} > u {candidate.score():.4f}).", member)
            return member

    def on_generation_end(self, generation : Generation, logger : Callable[[str], None]):
        self.memory.update()
        logger(f"SHADE: Average F-values: {average(self.F.values())}")
        logger(f"SHADE: Average Cr-values: {average(self.CR.values())}")

    def get_control_parameters(self) -> Tuple[float, float]:
        """
        The crossover probability CRi is generated according to a normal distribution
        of mean μCR and standard deviation 0.1 and then truncated to [0, 1].

        The mutation factor Fi is generated according to a Cauchy distribution
        with location parameter μF and scale parameter 0.1 and then
        truncated to be 1 if Fi >= 1 or regenerated if Fi <= 0.
        """
        # select random from memory
        r1 = random.randrange(0, self.memory.size)
        MF_i = self.memory.m_f[r1]
        MCR_i = self.memory.m_cr[r1]
        # generate MCR_i
        if MCR_i == None:
            CR_i = 0.0
        else:
            CR_i = clip(randn(MCR_i, 0.1), 0.0, 1.0)
        # generate MF_i
        while True:
            F_i = randc(MF_i, 0.1)
            if F_i <= self.F_MIN:
                continue
            if F_i >= self.F_MAX:
                F_i = self.F_MAX
                break
            break
        return CR_i, F_i
        
    def pbest_member(self, generation : Sequence[MemberState]) -> MemberState:
        """Sample a random top member from the popualtion."""
        sorted_members = sorted(generation, reverse=True)
        n_elitists = round(len(generation) * self.p)
        n_elitists = max(n_elitists, 1) # correction for too small p-values
        elitists = sorted_members[:n_elitists]
        return random.choice(elitists)

class LSHADE(SHADE):
    """
    A general, modifiable implementation of Success-History based Adaptive Differential Evolution (SHADE)
    with linear population size reduction.

    References:
        SHADE: https://ieeexplore.ieee.org/document/6557555
        L-SHADE: https://ieeexplore.ieee.org/document/6900380

    Parameters:
        N_INIT: The number of members in the population {15, 16, ..., 25}.
        r_arc: adjusts archive size with round(N_INIT * rarc) {1.0, 1.1, ..., 3.0}.
        MAX_NFE: the maximum number of fitness evaluations (N * (end_steps / step_size))
        p: control parameter for DE/current-to-pbest/1/. Small p, more greedily {0.05, 0.06, ..., 0.15}.
        memory_size: historical memory size (H) {2, 3, ..., 10}.
    """
    def __init__(self, N_INIT : int, MAX_NFE : int, r_arc : float = 2.0, p : float = 0.1, memory_size : int = 5, f_min : float = 0.0, f_max : float = 1.0) -> None:
        super().__init__(N_INIT, r_arc, p, memory_size, f_min, f_max)
        self.N_MIN = 4
        self.MAX_NFE = MAX_NFE
        self._nfe = 0

    def on_evaluate(self, candidates : Tuple[MemberState, MemberState], logger : Callable[[str], None]) -> MemberState:
        self._nfe += 1 # increment the number of fitness evaluations
        return super().on_evaluate(candidates, logger)

    def on_generation_end(self, generation : Generation, logger : Callable[[str], None]):
        self.adjust_generation_size(generation, logger)
        super().on_generation_end(generation, logger)

    def adjust_generation_size(self, generation : Generation, logger : Callable[[str], None]):
        new_size = round(((self.N_MIN - self.N_INIT) / self.MAX_NFE) * self._nfe + self.N_INIT)
        if new_size >= len(generation):
            return
        logger(f"adjusting generation size {len(generation)} --> {new_size}")
        # adjust archive size |A| according to |P|
        self.archive.size = new_size * self.r_arc
        size_delta = len(generation) - new_size
        for worst in sorted(generation)[:size_delta]:
            generation.remove(worst)
            logger(f"member {worst.id} with score {worst.score():.4f} was removed from the generation.")

def logistic(x : float, k : float = 20) -> float:
    return 1 / (1 + math.exp(-k * (x - 0.5)))

def curve(x : float, k : float = 5) -> float:
    return x**k

class DecayingLSHADE(LSHADE):
    """
    Decays the F-value by multiplying it with a guide. The guide can be a line, a box curve or a logistic curve.\n
    The guide starts at 1.0 and moves towards 0.0.
    """
    def __init__(self, N_INIT : int, MAX_NFE : int, r_arc : float = 2.0, p : float = 0.1, memory_size :int = 5, f_min : float = 0.0, f_max : float = 1.0, decay_type : str = 'linear') -> None:
        super().__init__(N_INIT, MAX_NFE, r_arc, p, memory_size, f_min, f_max)
        if decay_type == 'linear':
            self.decay_function = lambda f, nfe, max_nfe: f * (1.0 - nfe/max_nfe)
        elif decay_type == 'curve':
            self.decay_function = lambda f, nfe, max_nfe: f * (1.0 - curve(nfe/max_nfe))
        elif decay_type == 'logistic':
            self.decay_function = lambda f, nfe, max_nfe: f * (1.0 - logistic(nfe/max_nfe))
        else:
            raise NotImplementedError(f"'{decay_type}' is not implemented.'")

    def get_control_parameters(self) -> Tuple[float, float]:
        cr, f = super().get_control_parameters()
        return cr, self.decay_function(f, self._nfe, self.MAX_NFE)

class GuidedLSHADE(LSHADE):
    """
    Guides the F-value along a guide. The guide can be a line, a box curve or a logistic curve.\n
    The strength determines the guides influence on F. A strength of 1.0 perfectly maps it to the guide.
    """
    def __init__(self, N_INIT : int, MAX_NFE : int, r_arc : float = 2.0, p : float = 0.1, memory_size :int = 5, f_min : float = 0.0, f_max : float = 1.0, guide_type : str = 'linear', strength : int = 0.5) -> None:
        super().__init__(N_INIT, MAX_NFE, r_arc, p, memory_size, f_min, f_max)
        if guide_type == 'linear':
            self.guide_function = lambda f, nfe, max_nfe: f + ((1.0 - nfe/max_nfe) - f) * strength
        elif guide_type == 'curve':
            self.guide_function = lambda f, nfe, max_nfe: f + ((1.0 - curve(nfe/max_nfe)) - f) * strength
        elif guide_type == 'logistic':
            self.guide_function = lambda f, nfe, max_nfe: f + ((1.0 - logistic(nfe/max_nfe)) - f) * strength
        else:
            raise NotImplementedError(f"'{guide_type}' is not implemented.'")

    def get_control_parameters(self) -> Tuple[float, float]:
        cr, f = super().get_control_parameters()
        return cr, self.guide_function(f, self._nfe, self.MAX_NFE)
    
class LSHADEWithWeightSharing(LSHADE):
    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> Tuple[MemberState, MemberState]:
        """
        Perform crossover, mutation and selection according to the initial 'DE/current-to-pbest/1/bin'
        implementation of differential evolution, with adapted CR and F parameters. \n
        Also copies the network model- and optimizer state of the pbest member.
        """
        if len(generation) < 4:
            raise ValueError("generation size must be at least 4 or higher.")
        for index, member in generation.entries():
            # control parameter assignment
            self.CR[index], self.F[index] = self.get_control_parameters()
            # select random unique members from the union of the generation and archive
            x_r1, x_r2 = random_from_list(self.archive + list(generation), k=2, exclude=(member,))
            # select random best member
            x_pbest = self.pbest_member(generation)
            # hyper-parameter dimension size
            dimension_size = len(member.parameters)
            # choose random parameter dimension
            j_rand = random.randrange(0, dimension_size)
            # make a copy of the member
            candidate = member.copy()
            for j in range(dimension_size):
                if random.uniform(0.0, 1.0) <= self.CR[index] or j == j_rand:
                    mutant = de_current_to_best_1(F = self.F[index], x_base = member[j], x_best = x_pbest[j],
                        x_r1 = x_r1[j], x_r2 = x_r2[j])
                    mutant = halving(base = member[j], mutant = mutant,
                        lower_bounds = 0.0, upper_bounds = 1.0)
                    candidate[j] = mutant
                else:
                    candidate[j] = member[j]
            # copy weights from pbest
            existing = member.copy()
            existing.copy_state(x_pbest)
            candidate.copy_state(x_pbest)
            yield existing, candidate