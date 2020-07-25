import copy
import math
import heapq
import random
import warnings
from abc import abstractmethod
from typing import Tuple, Iterable, Sequence, Callable
from multiprocessing.managers import SyncManager

from pbt.utils.multiprocessing import Counter
from pbt.member import Checkpoint, Generation
from pbt.de.mutation import de_rand_1, de_current_to_best_1
from pbt.de.constraint import halving
from pbt.utils.constraint import clip
from pbt.utils.distribution import randn, randc, mean_wl
from pbt.utils.iterable import random_from_list


def best(members: Iterable[Checkpoint], n: int = 1) -> Sequence[Checkpoint]:
    return heapq.nlargest(n=n, iterable=members)

def worst(members: Iterable[Checkpoint], n: int = 1) -> Sequence[Checkpoint]:
    return heapq.nsmallest(n=n, iterable=members)


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
    def spawn(self, members: Iterable[Checkpoint]) -> Generation:
        """Create initial generation."""
        pass

    @abstractmethod
    def mutate(self, member: Checkpoint, generation: Generation, **kwargs) -> Checkpoint:
        """Called for each member in generation. Returns one candidate or multiple candidates."""
        pass


class DifferentialEvolveEngine(EvolveEngine):
    """
    Base class for all differential evolvers.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    @abstractmethod
    def spawn(self, members: Iterable[Checkpoint]) -> Generation:
        """Create initial generation."""
        pass

    @abstractmethod
    def on_generation_start(self, generation: Generation) -> None:
        """Called before each generation."""
        pass

    @abstractmethod
    def mutate(self, member: Checkpoint, generation: Generation, fitness_function: Callable[[Checkpoint], None]) -> Checkpoint:
        """Called for each member in generation. Returns one candidate or multiple candidates."""
        pass

    @abstractmethod
    def on_generation_end(self, generation: Generation) -> None:
        """Called at the end of each generation."""
        pass


class ExploitAndExplore(EvolveEngine):
    """
    A general, modifiable implementation of PBTs exploitation and exploration method.
    """

    PERTURB_METHODS = ('choice', 'sample')

    def __init__(self, exploit_factor: float = 0.2, explore_factors: Tuple[float, ...] = (0.8, 1.2), perturb_method: str = 'choice', **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(exploit_factor, float):
            raise TypeError(f"the 'exploit_factor' specified was of wrong type {type(exploit_factor)}, expected {float}.")
        if not(0.0 <= exploit_factor <= 1.0):
            raise ValueError(f"the 'exploit_factor' specified was not in range [0.0, 1.0].")
        if not isinstance(explore_factors, (list, tuple)):
            raise TypeError(f"the 'explore_factors' specified was of wrong type {type(explore_factors)}, expected {list} or {tuple}.")
        if not all(isinstance(factor, float) for factor in explore_factors):
            raise ValueError(f"the 'explore_factors' specified was not a sequence of floats.")
        if not isinstance(perturb_method, str):
            raise TypeError(f"the 'perturb_method' specified was of wrong type {type(perturb_method)}, expected {str}.")
        if perturb_method not in ExploitAndExplore.PERTURB_METHODS:
            raise NotImplementedError(f"perturb method '{perturb_method}' is not supported, expected {ExploitAndExplore.PERTURB_METHODS}.")
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors
        self.perturb_method = perturb_method

    def spawn(self, members: Iterable[Checkpoint]) -> Generation:
        """Create initial generation."""
        generation = Generation()
        for member in members:
            member = member.copy()
            [hp.sample_uniform() for hp in member.parameters]
            generation.append(member)
        return generation

    def mutate(self, member: Checkpoint, generation: Generation) -> Checkpoint:
        """
        Exploit best peforming members and explores all search spaces with random perturbation.
        A fraction of the bottom performing members exploit the top performing members.
        If member exploits, the hyper-parameters are parturbed.
        """
        if not isinstance(member, Checkpoint):
            raise TypeError(f"the 'member' specified was of wrong type {type(member)}, expected {Checkpoint}.")
        if not isinstance(generation, Generation):
            raise TypeError(f"the 'generation' specified was of wrong type {type(generation)}, expected {Generation}.")
        if len(generation) < 2:
            raise ValueError("generation size must be at least 2 or higher")
        if member not in generation:
            raise ValueError("member is required to be present in the specified generation")
        n_elitists = max(1, round(len(generation) * self.exploit_factor))
        elitists = best(generation, n_elitists)
        # exploit if member is not elitist
        if member not in elitists:
            elitist = random.choice(elitists)
            if not elitist.has_state():
                self.logger(f"member {member.uid} remains itself; elitist {elitist.uid} does not have state to share.")
                return member
            self.logger(f"member {member.uid} exploits and explores member {elitist.uid}...")
            member.copy_parameters(elitist)
            member.copy_state(elitist)
            member.copy_score(elitist)
            self.__explore(member)
            return member
        else:
            self.logger(f"member {member.uid} remains itself...")
            return member

    def __explore(self, member: Checkpoint):
        """Perturb all parameters by the defined explore_factors."""
        assert isinstance(member, Checkpoint)
        for parameter in member.parameters:
            perturb_factor = self._get_perturb_factor()
            parameter *= perturb_factor

    def _get_perturb_factor(self):
        if self.perturb_method == 'choice':
            return random.choice(self.explore_factors)
        elif self.perturb_method == 'sample':
            return random.uniform(self.explore_factors[0], self.explore_factors[1])
        else:
            raise NotImplementedError()


class DifferentialEvolution(DifferentialEvolveEngine):
    """
    A general, modifiable implementation of Differential Evolution (DE)
    """

    def __init__(self, F: float = 0.2, Cr: float = 0.8, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(F, float):
            raise TypeError(f"the 'F' specified was of wrong type {type(F)}, expected {float}.")
        if not isinstance(Cr, float):
            raise TypeError(f"the 'Cr' specified was of wrong type {type(Cr)}, expected {float}.")
        self.F = F
        self.Cr = Cr

    def on_generation_start(self, generation: Generation) -> None:
        pass

    def on_generation_end(self, generation: Generation) -> None:
        pass

    def spawn(self, members: Iterable[Checkpoint]) -> Generation:
        """Create initial generation."""
        generation = Generation()
        for member in members:
            member = member.copy()
            [hp.sample_uniform() for hp in member.parameters]
            generation.append(member)
        return generation

    def mutate(self, parent: Checkpoint, generation: Generation, fitness_function: Callable[[Checkpoint], None]) -> Checkpoint:
        """
        Perform crossover, mutation and selection according to the initial 'DE/rand/1/bin'
        implementation of differential evolution.
        """
        if not isinstance(parent, Checkpoint):
            raise TypeError(f"the 'parent' specified was of wrong type {type(parent)}, expected {Checkpoint}.")
        if not isinstance(generation, Generation):
            raise TypeError(f"the 'generation' specified was of wrong type {type(generation)}, expected {Generation}.")
        if len(generation) < 3:
            raise ValueError("generation size must be at least 3 or higher")
        if parent not in generation:
            raise ValueError("parent is required to be present in the specified generation")
        if not callable(fitness_function):
            raise ValueError("fitness_function is not callable")
        # copy parent
        parent = parent.copy()
        dimensions = len(parent.parameters)
        x_r0, x_r1, x_r2 = random_from_list(generation, k=3, exclude=(parent,))
        j_rand = random.randrange(0, dimensions)
        self._print_mutation_parameters(parent=parent, x_r0=x_r0, x_r1=x_r1, x_r2=x_r2, j_rand=j_rand)
        self.logger(f"generating trial member {parent.uid}")
        trial = parent.copy()
        for j in range(dimensions):
            CR_ri = random.uniform(0.0, 1.0)
            if CR_ri <= self.Cr or j == j_rand:
                self.logger(f"M{parent.uid}: crossover in dimension {j} with CR_ri {CR_ri:.4f}")
                mutant = de_rand_1(F=self.F, x_r0=x_r0[j], x_r1=x_r1[j], x_r2=x_r2[j])
                constrained = clip(mutant, 0.0, 1.0)
                trial[j] = constrained
                self.logger(f"M{parent.uid}: mutant value {mutant:.4f}, constrained to {constrained:.4f}")
            else:
                trial[j] = parent[j]
        # measure fitness
        self.logger(f"M{parent.uid}: measuring fitness score of parent and trial")
        fitness_function(parent)
        fitness_function(trial)
        # select best
        self.logger(f"M{parent.uid}: selecting between evaluated parent and trial")
        return self._select(parent, trial)

    def _select(self, parent: Checkpoint, trial: Checkpoint) -> Checkpoint:
        """Evaluates candidate, compares it to the base and returns the best performer."""
        if parent <= trial:
            self.logger(f"M{parent.uid}: mutate member (x {parent.eval_score():.4f} <= u {trial.eval_score():.4f}).")
            return trial
        else:
            self.logger(f"M{parent.uid}: maintain member (x {parent.eval_score():.4f} > u {trial.eval_score():.4f}).")
            return parent
    
    def _print_mutation_parameters(self, parent, x_r0, x_r1, x_r2, j_rand):
        if not self.verbose:
            return
        lines = [
            f"M{parent.uid} mutation parameters:",
            f"control parameters: CR {self.Cr:.4f}, F {self.F:.4f}",
            f"x_r0: {x_r0} with score {x_r0.eval_score()}",
            f"x_r1: {x_r1} with score {x_r1.eval_score()}",
            f"x_r2: {x_r2} with score {x_r2.eval_score()}",
            f"random crossover dimension (j_rand): {j_rand}"]
        text = '\n\t'.join(lines)
        self.logger(text)


class HistoricalMemory(object):
    def __init__(self, manager: SyncManager, size: int, default: float = 0.5) -> None:
        if not isinstance(manager, SyncManager):
            raise TypeError(f"the 'manager' specified was of wrong type {type(manager)}, expected {SyncManager}.")
        if not isinstance(size, int):
            raise TypeError(f"the 'size' specified was of wrong type {type(size)}, expected {int}.")
        if not isinstance(default, float):
            raise TypeError(f"the 'default' specified was of wrong type {type(default)}, expected {float}.")
        self.size = size
        self.m_cr = [default] * size
        self.m_f = [default] * size
        self.__lock = manager.Lock()
        self.__s_cr = manager.list()
        self.__s_f = manager.list()
        self.__s_w = manager.list()
        self.__k = 0

    def record(self, cr: float, f: float, w: float) -> None:
        """Save control parameters and delta score (weight) to historical memory."""
        if not isinstance(cr, float):
            raise TypeError(f"the 'cr' specified was of wrong type {type(cr)}, expected {float}.")
        if not math.isfinite(cr) or not (0.0 <= cr <= 1.0):
            raise ValueError(f"cr value {cr} is not valid.")
        if not isinstance(f, float):
            raise TypeError(f"the 'f' specified was of wrong type {type(f)}, expected {float}.")
        if not math.isfinite(f) or f < 0.0:
            raise ValueError(f"f value {f} is not valid.")
        if not isinstance(w, float):
            raise TypeError(f"the 'w' specified was of wrong type {type(w)}, expected {float}.")
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(f"w value {w} is not valid.")
        with self.__lock:
            self.__s_cr.append(cr)
            self.__s_f.append(f)
            self.__s_w.append(w)

    def reset(self) -> None:
        """Reset S_CR, S_F and weights to empty lists."""
        with self.__lock:
            self.__s_cr[:] = []
            self.__s_f[:] = []
            self.__s_w[:] = []

    def update(self) -> None:
        with self.__lock:
            assert len(self.__s_cr) == len(self.__s_f) == len(self.__s_w), "the lengths of __s_cr, __s_f and __s_weights are not equal."
            if len(self.__s_cr) == 0:
                return
            if self.m_cr[self.__k] == None or max(self.__s_cr) == 0.0:
                self.m_cr[self.__k] = None
            else:
                self.m_cr[self.__k] = mean_wl(self.__s_cr, self.__s_w)
            self.m_f[self.__k] = mean_wl(self.__s_f, self.__s_w)
            self.__k = 0 if self.__k >= self.size - 1 else self.__k + 1


class ExternalArchive():
    def __init__(self, manager: SyncManager, size: int, verbose: bool = False) -> None:
        if not isinstance(manager, SyncManager):
            raise TypeError(f"the 'manager' specified was of wrong type {type(manager)}, expected {SyncManager}.")
        if not isinstance(size, int):
            raise TypeError(f"the 'size' specified was of wrong type {type(size)}, expected {int}.")
        if not isinstance(verbose, bool):
            raise TypeError(f"the 'verbose' specified was of wrong type {type(verbose)}, expected {bool}.")
        self.__lock = manager.Lock()
        self.__size = manager.Value('i', size)
        self.__records = manager.list()
        self.__verbose = verbose

    @property
    def records(self):
        with self.__lock:
            return list(self.__records)

    def __print(self, message: str):
        if not self.__verbose:
            return
        print(f"ExternalArchive: {message}")

    def __random_delete(self, n: int = 1):
        assert n >= 0, "n is negative!"
        assert n <= len(self.__records), "attempted to remove too many values"
        for _ in range(n):
            random_value = random.choice(self.__records)
            index = self.__records.index(random_value)
            self.__print(f"removing random {random_value} at index {index} from archive.")
            self.__records.remove(random_value)

    def resize(self, size: int):
        if not isinstance(size, int):
            raise TypeError(f"the 'size' specified was of wrong type {type(size)}, expected {int}.")
        if size < 0:
            raise ValueError("specified size is negative")
        if size > self.__size.value:
            raise ValueError("specified size is larger than current size")
        with self.__lock:
            # assign new size value
            self.__size.value = size
            # randomly remove overflow
            overflow = len(self.__records) - self.__size.value
            if overflow > 0:
                self.__random_delete(n=overflow)

    def append(self, parent: Checkpoint) -> None:
        if not isinstance(parent, Checkpoint):
            raise TypeError(f"the 'parent' specified was of wrong type {type(parent)}, expected {Checkpoint}.")
        if parent in self.__records:
            raise ValueError("checkpoint already exists in archive.")
        with self.__lock:
            self.__print(f"appending {parent} to archive of size {len(self.__records)}")
            if len(self.__records) == self.__size.value:
                self.__random_delete(n=1)
            parent.delete_state() # remove useless state
            self.__records.append(parent)

    def clear(self) -> None:
        """Clear records"""
        with self.__lock:
            self.__records[:] = []


class SHADE(DifferentialEvolveEngine):
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

    def __init__(self, manager, N_INIT: int, r_arc: float = 2.0, p: float = 0.1, memory_size: int = 5, f_min: float = 0.0, f_max: float = 1.0, state_sharing: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(manager, SyncManager):
            raise TypeError(f"the 'manager' specified was of wrong type {type(manager)}, expected {SyncManager}.")
        if not isinstance(N_INIT, int):
            raise TypeError(f"the 'N_INIT' specified was of wrong type {type(N_INIT)}, expected {int}.")
        if N_INIT < 4:
            raise ValueError("the 'N_INIT' specified must be at least 4 or higher.")
        if not isinstance(r_arc, float):
            raise TypeError(f"the 'r_arc' specified was of wrong type {type(r_arc)}, expected {float}.")
        if not isinstance(p, float):
            raise TypeError(f"the 'p' specified was of wrong type {type(p)}, expected {float}.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("the 'p' specified was not in range [0.0, 1.0].")
        if round(N_INIT * p) < 1:
            warnings.warn(f"the 'p' specified was too low for the provided population size. It must be atleast {1.0 / N_INIT} for population size of {N_INIT}. This will be resolved by always choosing the top one performer in the population as pbest.")
        if not isinstance(memory_size, int):
            raise TypeError(f"the 'memory_size' specified was of wrong type {type(memory_size)}, expected {int}.")
        if memory_size < 0.0:
            raise ValueError("the 'memory_size' specified was negative.")
        if not isinstance(f_min, float):
            raise TypeError(f"the 'f_min' specified was of wrong type {type(f_min)}, expected {float}.")
        if f_min < 0.0:
            raise ValueError("the 'f_min' specified was negative.")
        if not isinstance(f_max, float):
            raise TypeError(f"the 'f_max' specified was of wrong type {type(f_max)}, expected {float}.")
        if f_max < 0.0:
            raise ValueError("the 'f_max' specified was negative.")
        if f_max < f_min:
            raise ValueError("the 'f_max' specified was less than 'f_min'.")
        if not isinstance(state_sharing, bool):
            raise TypeError(f"the 'state_sharing' specified was of wrong type {type(state_sharing)}, expected {bool}.")
        self.archive = ExternalArchive(
            manager=manager, size=round(N_INIT * r_arc), verbose=self.verbose)
        self.memory = HistoricalMemory(
            manager=manager, size=memory_size, default=(f_max - f_min) / 2.0)
        self.F_MIN = f_min
        self.F_MAX = f_max
        self.N_INIT = N_INIT
        self.r_arc = r_arc
        self.p = p
        self.state_sharing = state_sharing

    def spawn(self, members: Iterable[Checkpoint]) -> Generation:
        """Create initial generation."""
        generation = Generation()
        for member in members:
            member = member.copy()
            [hp.sample_uniform() for hp in member.parameters]
            generation.append(member)
        return generation

    def on_generation_start(self, generation: Generation) -> None:
        if not isinstance(generation, Generation):
            raise TypeError(f"the 'generation' specified was of wrong type {type(generation)}, expected {Generation}.")
        self.memory.reset()

    def mutate(self, parent: Checkpoint, generation: Generation, fitness_function: Callable[[Checkpoint], None]) -> Checkpoint:
        """
        Perform crossover, mutation and selection according to the initial 'DE/current-to-pbest/1/bin'
        implementation of differential evolution, with adapted CR and F parameters.
        """
        if not isinstance(parent, Checkpoint):
            raise TypeError(f"the 'parent' specified was of wrong type {type(parent)}, expected {Checkpoint}.")
        if not isinstance(generation, Generation):
            raise TypeError(f"the 'generation' specified was of wrong type {type(generation)}, expected {Generation}.")
        if len(generation) < 4:
            raise ValueError("generation size must be at least 4 or higher")
        if parent not in generation:
            raise ValueError("parent is required to be present in the specified generation")
        if not callable(fitness_function):
            raise ValueError("fitness_function is not callable")
        # copy parent
        parent = parent.copy()
        # control parameter assignment
        CR_i, F_i = self._get_control_parameters()
        # select random unique members from the union of the generation and archive
        x_r1, x_r2 = self._sample_r1_and_r2(parent, generation)
        # select random best member
        x_pbest = self._sample_pbest_member(generation)
        # hyper-parameter dimension size
        dimensions = len(parent.parameters)
        # choose random parameter dimension
        j_rand = random.randrange(0, dimensions)
        self._print_mutation_parameters(parent=parent, CR_i=CR_i, F_i=F_i, x_r1=x_r1, x_r2=x_r2, x_pbest=x_pbest, j_rand=j_rand)
        # make a copy of the member
        trial = parent.copy()
        if self.state_sharing:
            self.logger(f"M{parent.uid}: copying state from x_pbest member {x_pbest.uid}")
            trial.copy_state(x_pbest)
        self.logger(f"M{parent.uid}: generating trial member")
        for j in range(dimensions):
            CR_ri = random.uniform(0.0, 1.0)
            if CR_ri <= CR_i or j == j_rand:
                self.logger(f"M{parent.uid}: crossover in dimension {j} with CR_ri {CR_ri:.4f}")
                mutant = de_current_to_best_1(F=F_i, x_base=parent[j], x_best=x_pbest[j], x_r1=x_r1[j], x_r2=x_r2[j])
                constrained = halving(base=parent[j], mutant=mutant, lower_bounds=0.0, upper_bounds=1.0)
                trial[j] = constrained
                self.logger(f"M{parent.uid}: mutant value {mutant:.4f}, constrained to {constrained:.4f}")
            else:
                trial[j] = parent[j]
        # measure fitness
        self.logger(f"M{parent.uid}: measuring fitness score of parent and trial")
        fitness_function(parent)
        fitness_function(trial)
        # select
        self.logger(f"M{parent.uid}: selecting between measured parent and trial")
        return self._select(parent, trial, CR_i, F_i)

    def _select(self, parent: Checkpoint, trial: Checkpoint, CR_i: float, F_i: float) -> Checkpoint:
        """Evaluates candidate, compares it to the original member and returns the best performer."""
        if parent <= trial:
            if parent < trial:
                self.logger(f"M{parent.uid}: adding parent to archive.")
                self.archive.append(parent.copy())
                w_i = abs(trial.eval_score() - parent.eval_score())
                self.logger(f"M{parent.uid}: recording CR_i {CR_i:.4f} and F_i {F_i:.4f} with w_i {w_i:.4E} to historical memory.")
                self.memory.record(CR_i, F_i, w_i)
            self.logger(f"M{parent.uid}: mutate member (x {parent.eval_score():.4f} < u {trial.eval_score():.4f}).")
            return trial
        else:
            self.logger(
                f"M{parent.uid}: maintain member (x {parent.eval_score():.4f} > u {trial.eval_score():.4f}).")
            return parent

    def on_generation_end(self, generation: Generation):
        if not isinstance(generation, Generation):
            raise TypeError(f"the 'generation' specified was of wrong type {type(generation)}, expected {Generation}.")
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
        assert not math.isnan(MF_i), "MF_i is NaN."
        assert not math.isnan(MCR_i), "MCR_i is NaN."
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

    def _sample_r1_and_r2(self, member: Checkpoint, generation: Generation) -> Tuple[Checkpoint, Checkpoint]:
        x_r1 = random_from_list(list(generation), k=1, exclude=(member,))
        x_r2 = random_from_list(self.archive.records + list(generation), k=1, exclude=(member, x_r1))
        return x_r1, x_r2

    def _sample_pbest_member(self, generation: Generation) -> Checkpoint:
        """Sample a random top member from the popualtion."""
        n_elitists = max(1, round(len(generation) * self.p))
        elitists = best(generation, n_elitists)
        return random.choice(elitists)

    def _print_mutation_parameters(self, parent, CR_i, F_i, x_r1, x_r2, x_pbest, j_rand):
        if not self.verbose:
            return
        lines = [
            f"M{parent.uid} mutation parameters:",
            f"control parameters: CR_i {CR_i:.4f}, F_i {F_i:.4f}",
            f"x_r1: {x_r1} with score {x_r1.eval_score()}",
            f"x_r2: {x_r2} with score {x_r2.eval_score()}",
            f"x_pbest: {x_pbest} with score {x_pbest.eval_score()}",
            f"random crossover dimension (j_rand): {j_rand}"]
        text = '\n\t'.join(lines)
        self.logger(text)


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

    def __init__(self, manager: SyncManager, MAX_NFE: int, **kwargs) -> None:
        super().__init__(manager, **kwargs)
        if not isinstance(MAX_NFE, int):
            raise TypeError(f"the 'MAX_NFE' specified was of wrong type {type(MAX_NFE)}, expected {int}.")
        self.N_MIN = 4
        self.MAX_NFE = MAX_NFE
        self._nfe = Counter(manager=manager, value=0)

    def _select(self, parent: Checkpoint, trial: Checkpoint, CR_i: float, F_i: float) -> Checkpoint:
        self._nfe.increment()  # increment the number of fitness evaluations
        return super()._select(parent, trial, CR_i, F_i)

    def on_generation_end(self, generation: Generation):
        super().on_generation_end(generation)
        self._adjust_generation_size(generation)

    def _adjust_generation_size(self, generation: Generation):
        new_size = round(((self.N_MIN - self.N_INIT) / self.MAX_NFE) * self._nfe.value + self.N_INIT)
        if new_size >= len(generation):
            return
        self.logger(
            f"adjusting generation size {len(generation)} --> {new_size}")
        # adjust archive size |A| according to |P|
        self.archive.resize(round(new_size * self.r_arc))
        # remove Delta-N worst members
        size_delta = len(generation) - new_size
        for member in worst(generation, size_delta):
            generation.remove(member)
            self.logger(
                f"member {member.uid} with score {member.eval_score():.4f} was removed from the generation.")


def logistic(x: float, k: float = 20) -> float:
    return 1 / (1 + math.exp(-k * (x - 0.5)))


def curve(x: float, k: float = 5) -> float:
    return x**k


class DecayingLSHADE(LSHADE):
    """
    Decays the F-value by multiplying it with a guide. The guide can be a line, a box curve or a logistic curve.\n
    The guide starts at 1.0 and moves towards 0.0.
    """

    def __init__(self, decay_type: str = 'linear', **kwargs) -> None:
        super().__init__(**kwargs)
        if decay_type == 'linear':
            self.decay_function = lambda f, nfe, max_nfe: f * \
                (1.0 - nfe/max_nfe)
        elif decay_type == 'curve':
            self.decay_function = lambda f, nfe, max_nfe: f * \
                (1.0 - curve(nfe/max_nfe))
        elif decay_type == 'logistic':
            self.decay_function = lambda f, nfe, max_nfe: f * \
                (1.0 - logistic(nfe/max_nfe))
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

    def __init__(self, guide_type: str = 'linear', strength: int = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        if guide_type == 'linear':
            self.guide_function = lambda f, nfe, max_nfe: f + \
                ((1.0 - nfe/max_nfe) - f) * strength
        elif guide_type == 'curve':
            self.guide_function = lambda f, nfe, max_nfe: f + \
                ((1.0 - curve(nfe/max_nfe)) - f) * strength
        elif guide_type == 'logistic':
            self.guide_function = lambda f, nfe, max_nfe: f + \
                ((1.0 - logistic(nfe/max_nfe)) - f) * strength
        else:
            raise NotImplementedError(f"'{guide_type}' is not implemented.'")

    def _get_control_parameters(self) -> Tuple[float, float]:
        cr, f = super()._get_control_parameters()
        return cr, self.guide_function(f, self._nfe.value, self.MAX_NFE)
