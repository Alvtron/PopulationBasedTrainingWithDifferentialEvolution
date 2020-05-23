import math
import random
import copy
import warnings
import collections
from abc import abstractmethod
from typing import Tuple, List, Dict, Iterable, Sequence, Callable, Generator

from pbt.utils.multiprocessing import Counter

from .member import MemberState, Generation
from .hyperparameters import Hyperparameters
from .de.mutation import de_rand_1, de_current_to_best_1
from .de.constraint import halving
from .utils.constraint import clip
from .utils.distribution import randn, randc, mean_wl
from .utils.iterable import grid, random_from_list, average

Candidates = Tuple[MemberState, ...]

class EvolveEngine(object):
    """
    Base class for all evolvers.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def logger(self, text: str) -> None:
        if not self.verbose:
            return
        print(text)
        
    @abstractmethod
    def spawn(self, members : Iterable[MemberState], **kwargs) -> Generation:
        """Create initial generation."""
        pass
    
    @abstractmethod
    def on_generation_start(self, generation : Generation, **kwargs) -> None:
        """Called before each generation."""
        pass
    
    @abstractmethod
    def mutate(self, member: MemberState, **kwargs) -> MemberState:
        """Called for each member in generation. Returns one candidate or multiple candidates."""
        pass

    @abstractmethod
    def on_generation_end(self, generation : Generation, **kwargs) -> None:
        """Called at the end of each generation."""
        pass

class ExploitAndExplore(EvolveEngine):
    """
    A general, modifiable implementation of PBTs exploitation and exploration method.
    """
    def __init__(self, exploit_factor: float = 0.2, explore_factors: Tuple[float, ...] = (0.8, 1.2), perturb_method: str = 'choice', **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(exploit_factor, float) and 0.0 <= exploit_factor <= 1.0, f"Exploit factor must be of type {float} between 0.0 and 1.0."
        assert isinstance(explore_factors, (float, list, tuple)), f"Explore factors must be of type {float}, {tuple} or {list}."
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors
        self.perturb_method = perturb_method

    def spawn(self, members : Iterable[MemberState]) -> Generation:
        """Create initial generation."""
        generation = Generation()
        for member in members:
            member = member.copy()
            [hp.sample_uniform() for hp in member.parameters]
            generation.append(member)
        return generation
    
    def mutate(self, member: MemberState, generation:Generation) -> MemberState:
        """
        Exploit best peforming members and explores all search spaces with random perturbation.
        A fraction of the bottom performing members exploit the top performing members.
        If member exploits, the hyper-parameters are parturbed.
        """
        n_elitists = max(1, round(len(generation) * self.exploit_factor))
        sorted_members = sorted(generation, reverse=True)
        elitists = sorted_members[:n_elitists]
        # exploit if member is not elitist
        if member not in elitists:
            elitist = random.choice(elitists)
            if not elitist.has_state():
                self.logger(f"member {member.id} remains itself; elitist {elitist.id} does not have state to share.")
                return member.copy()
            exploiter = member.copy()
            self.logger(f"member {exploiter.id} exploits member {elitist.id}...")
            exploiter.copy_parameters(elitist)
            exploiter.copy_state(elitist)
            exploiter.copy_score(elitist)
            self.logger(f"member {exploiter.id} explores member {elitist.id}...")
            return self.__explore(exploiter)
        else:
            self.logger(f"member {member.id} remains itself...")
            return member.copy()

    def __explore(self, member: MemberState) -> MemberState:
        """Perturb all parameters by the defined explore_factors."""
        explorer = member.copy()
        for index, _ in enumerate(explorer.parameters):
            perturb_factor = self._get_perturb_factor()
            explorer[index] = explorer[index] * perturb_factor
        return explorer

    def _get_perturb_factor(self):
        if self.perturb_method == 'choice':
            return random.choice(self.explore_factors)
        elif self.perturb_method == 'sample':
            return random.uniform(self.explore_factors[0], self.explore_factors[1])
        else:
            raise NotImplementedError()


class DifferentialEvolution(EvolveEngine):
    """
    A general, modifiable implementation of Differential Evolution (DE)
    """
    def __init__(self, F: float = 0.2, Cr: float = 0.8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.F = F
        self.Cr = Cr

    def spawn(self, members : Iterable[MemberState]) -> Generation:
        """Create initial generation."""
        generation = Generation()
        for member in members:
            member = member.copy()
            [hp.sample_uniform() for hp in member.parameters]
            generation.append(member)
        return generation

    def mutate(self, parent: MemberState, generation : Generation, fitness_function: Callable[[MemberState], None]) -> MemberState:
        """
        Perform crossover, mutation and selection according to the initial 'DE/rand/1/bin'
        implementation of differential evolution.
        """
        if len(generation) < 3:
            raise ValueError("generation size must be at least 3 or higher.")
        hp_dimension_size = len(parent.parameters)
        x_r0, x_r1, x_r2 = random_from_list(generation, k=3, exclude=(parent,))
        j_rand = random.randrange(0, hp_dimension_size)
        self.logger(f"generating trial member {parent.id}")
        trial = parent.copy()
        for j in range(hp_dimension_size):
            if random.uniform(0.0, 1.0) <= self.Cr or j == j_rand:
                mutant = de_rand_1(F = self.F, x_r0 = x_r0[j], x_r1 = x_r1[j], x_r2 = x_r2[j])
                constrained = clip(mutant, 0.0, 1.0)
                trial[j] = constrained 
            else:
                trial[j] = parent[j]
        # measure fitness
        self.logger(f"measuring fitness score of parent- and trial member {parent.id}")
        fitness_function(parent)
        fitness_function(trial)
        # select best
        self.logger(f"selecting between evaluated parent- and trial member {parent.id}")
        return self._select(parent, trial)

    def _select(self, parent: MemberState, trial: MemberState) -> MemberState:
        """Evaluates candidate, compares it to the base and returns the best performer."""
        if parent <= trial :
            self.logger(f"mutate member {parent.id} (x {parent.eval_score():.4f} <= u {trial.eval_score():.4f}).")
            return trial
        else:
            self.logger(f"maintain member {parent.id} (x {parent.eval_score():.4f} > u {trial.eval_score():.4f}).")
            return parent

class HistoricalMemory(object):
    def __init__(self, manager, size : int, default : float = 0.5) -> None:
        self.size = size
        self.m_cr = [default] * size
        self.m_f = [default] * size
        self.s_cr = manager.list()
        self.s_f = manager.list()
        self.s_w = manager.list()
        self.k = 0

    def record(self, cr_i: float, f_i: float, delta_score: float) -> None:
        """Save control parameters and delta score (weight) to historical memory."""
        assert 0.0 <= cr_i <= 1.0, "cr_i is not valid."
        assert 0.0 <= f_i, "f_i is not valid."
        if delta_score == 0.0 or math.isnan(delta_score):
            # set delta score extremely small to avoid division by zero
            delta_score = 1e-9
        self.s_cr.append(cr_i)
        self.s_f.append(f_i)
        self.s_w.append(delta_score)
    
    def reset(self) -> None:
        """Reset S_CR, S_F and weights to empty lists."""
        self.s_cr[:] = []
        self.s_f[:] = []
        self.s_w[:] = []

    def update(self) -> None:
        assert len(self.s_cr) == len(self.s_f) == len(self.s_w), "the lengths of s_cr, s_f and s_weights are not equal."
        if len(self.s_cr) == 0:
            return
        if self.m_cr[self.k] == None or max(self.s_cr) == 0.0:
            self.m_cr[self.k] = None
        else:
            self.m_cr[self.k] = mean_wl(self.s_cr, self.s_w)
        self.m_f[self.k] = mean_wl(self.s_f, self.s_w)
        self.k = 0 if self.k >= self.size - 1 else self.k + 1

class ExternalArchive():
    def __init__(self, manager, size : int) -> None:
        self.size = size
        self.records = manager.list()

    def append(self, parent: MemberState) -> None:
        if len(self.records) == self.size:
            random_value = random.choice(self.records)
            self.records.remove(random_value)
        self.records.append(parent)

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
    def __init__(self, manager, N_INIT : int, r_arc : float = 2.0, p : float = 0.1, memory_size : int = 5, f_min : float = 0.0, f_max : float = 1.0, state_sharing: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        if N_INIT < 4:
            raise ValueError("population size must be at least 4 or higher.")
        if round(N_INIT * p) < 1:
            warnings.warn(f"p-parameter too low for the provided population size. It must be atleast {1.0 / N_INIT} for population size of {N_INIT}. This will be resolved by always choosing the top one performer in the population as pbest.")
        if f_min < 0:
            raise ValueError("f_min cannot be negative.")
        if f_max <= 0:
            raise ValueError("f_max cannot be negative.")
        if f_max < f_min:
            raise ValueError("f_max cannot be less than f_min.")
        self.archive = ExternalArchive(manager=manager, size=round(N_INIT * r_arc))
        self.memory = HistoricalMemory(manager=manager, size=memory_size, default=(f_max - f_min) / 2.0)
        self.F_MIN = f_min
        self.F_MAX = f_max
        self.N_INIT = N_INIT
        self.r_arc = r_arc
        self.p = p
        self.state_sharing = state_sharing

    def spawn(self, members : Iterable[MemberState]) -> Generation:
        """Create initial generation."""
        generation = Generation()
        for member in members:
            member = member.copy()
            [hp.sample_uniform() for hp in member.parameters]
            generation.append(member)
        return generation

    def on_generation_start(self, generation: Generation) -> None:
        self.memory.reset()

    def mutate(self, parent: MemberState, generation: Generation, fitness_function: Callable[[MemberState], None]) -> MemberState:
        """
        Perform crossover, mutation and selection according to the initial 'DE/current-to-pbest/1/bin'
        implementation of differential evolution, with adapted CR and F parameters.
        """
        if len(generation) < 4:
            raise ValueError("generation size must be at least 4 or higher.")
        # control parameter assignment
        CR_i, F_i = self._get_control_parameters()
        # select random unique members from the union of the generation and archive
        x_r1, x_r2 = self._sample_r1_and_r2(parent, generation)
        # select random best member
        x_pbest = self._sample_pbest_member(generation)
        # hyper-parameter dimension size
        dimension_size = len(parent.parameters)
        # choose random parameter dimension
        j_rand = random.randrange(0, dimension_size)
        # make a copy of the member
        self.logger(f"generating trial member {parent.id}")
        trial = parent.copy()
        if self.state_sharing:
            trial.copy_state(x_pbest)
        for j in range(dimension_size):
            if random.uniform(0.0, 1.0) <= CR_i or j == j_rand:
                mutant = de_current_to_best_1(F = F_i, x_base = parent[j],
                    x_best = x_pbest[j], x_r1 = x_r1[j], x_r2 = x_r2[j])
                constrained = halving(base = parent[j], mutant = mutant,
                    lower_bounds = 0.0, upper_bounds = 1.0)
                trial[j] = constrained
            else:
                trial[j] = parent[j]
        # measure fitness
        self.logger(f"measuring fitness score of parent/trial member {parent.id}")
        fitness_function(parent)
        fitness_function(trial)
        # select
        self.logger(f"selecting between measured parent/trial member {parent.id}")
        return self._select(parent, trial, CR_i, F_i)

    def _select(self, parent: MemberState, trial: MemberState, CR_i: float, F_i: float) -> MemberState:
        """Evaluates candidate, compares it to the original member and returns the best performer."""
        if parent <= trial:
            self.logger(f"For member {parent.id}: adding parent member {parent.id} to archive.")
            self.archive.append(parent.copy())
            delta_score = abs(trial.eval_score() - parent.eval_score())
            self.logger(f"For member {parent.id}: recording CR_i {CR_i:.3f} and F_i {F_i:.3f} with delta score {delta_score:.3f} to historical memory.")
            self.memory.record(CR_i, F_i, delta_score)
            self.logger(f"mutate member {parent.id} (x {parent.eval_score():.4f} < u {trial.eval_score():.4f}).")
            return trial
        else:
            self.logger(f"maintain member {parent.id} (x {parent.eval_score():.4f} > u {trial.eval_score():.4f}).")
            return parent

    def on_generation_end(self, generation: Generation):
        self.memory.update()

    def _get_control_parameters(self) -> Tuple[float, float]:
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
        assert not math.isnan(MF_i) , "MF_i is NaN."
        assert not math.isnan(MCR_i) , "MCR_i is NaN."
        # generate MCR_i
        if MCR_i == None:
            CR_i = 0.0
        else:
            CR_i = clip(randn(MCR_i, 0.1), 0.0, 1.0)
        # generate MF_i
        while True:
            F_i = randc(MF_i, 0.1)
            if F_i < self.F_MIN:
                continue
            if F_i > self.F_MAX:
                F_i = self.F_MAX
            break
        return CR_i, F_i
    
    def _sample_r1_and_r2(self, member: MemberState, generation: Generation) -> Tuple[MemberState, MemberState]:
        x_r1 = random_from_list(list(generation), k=1, exclude=(member,))
        x_r2 = random_from_list(list(self.archive.records) + list(generation), k=1, exclude=(member, x_r1))
        return x_r1, x_r2

    def _sample_pbest_member(self, generation: Generation) -> MemberState:
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
        MAX_NFE: The maximum number of fitness evaluations to peform.
        + SHADE parameters
    """
    def __init__(self, MAX_NFE : int, manager, **kwargs) -> None:
        super().__init__(manager, **kwargs)
        self.N_MIN = 4
        self.MAX_NFE = MAX_NFE
        self._nfe = Counter(manager=manager, value=0)

    def _select(self, parent: MemberState, trial: MemberState, CR_i: float, F_i: float) -> MemberState:
        self._nfe.increment() # increment the number of fitness evaluations
        return super()._select(parent, trial, CR_i, F_i)

    def on_generation_end(self, generation : Generation):
        self._adjust_generation_size(generation)
        super().on_generation_end(generation)

    def _adjust_generation_size(self, generation : Generation):
        new_size = round(((self.N_MIN - self.N_INIT) / self.MAX_NFE) * self._nfe.value + self.N_INIT)
        if new_size >= len(generation):
            return
        self.logger(f"adjusting generation size {len(generation)} --> {new_size}")
        # adjust archive size |A| according to |P|
        self.archive.size = new_size * self.r_arc
        size_delta = len(generation) - new_size
        for worst in sorted(generation)[:size_delta]:
            generation.remove(worst)
            self.logger(f"member {worst.id} with score {worst.eval_score():.4f} was removed from the generation.")

def logistic(x : float, k : float = 20) -> float:
    return 1 / (1 + math.exp(-k * (x - 0.5)))

def curve(x : float, k : float = 5) -> float:
    return x**k

class DecayingLSHADE(LSHADE):
    """
    Decays the F-value by multiplying it with a guide. The guide can be a line, a box curve or a logistic curve.\n
    The guide starts at 1.0 and moves towards 0.0.
    """
    def __init__(self, decay_type : str = 'linear', **kwargs) -> None:
        super().__init__(**kwargs)
        if decay_type == 'linear':
            self.decay_function = lambda f, nfe, max_nfe: f * (1.0 - nfe/max_nfe)
        elif decay_type == 'curve':
            self.decay_function = lambda f, nfe, max_nfe: f * (1.0 - curve(nfe/max_nfe))
        elif decay_type == 'logistic':
            self.decay_function = lambda f, nfe, max_nfe: f * (1.0 - logistic(nfe/max_nfe))
        else:
            raise NotImplementedError(f"'{decay_type}' is not implemented.'")

    def _get_control_parameters(self) -> Tuple[float, float]:
        cr, f = super()._get_control_parameters()
        return cr, self.decay_function(f, self._nfe.value, self.MAX_NFE)

class GuidedLSHADE(LSHADE):
    """
    Guides the F-value along a guide. The guide can be a line, a box curve or a logistic curve.\n
    The strength determines the guides influence on F. A strength of 1.0 perfectly maps it to the guide.
    """
    def __init__(self, guide_type : str = 'linear', strength : int = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        if guide_type == 'linear':
            self.guide_function = lambda f, nfe, max_nfe: f + ((1.0 - nfe/max_nfe) - f) * strength
        elif guide_type == 'curve':
            self.guide_function = lambda f, nfe, max_nfe: f + ((1.0 - curve(nfe/max_nfe)) - f) * strength
        elif guide_type == 'logistic':
            self.guide_function = lambda f, nfe, max_nfe: f + ((1.0 - logistic(nfe/max_nfe)) - f) * strength
        else:
            raise NotImplementedError(f"'{guide_type}' is not implemented.'")

    def _get_control_parameters(self) -> Tuple[float, float]:
        cr, f = super()._get_control_parameters()
        return cr, self.guide_function(f, self._nfe.value, self.MAX_NFE)