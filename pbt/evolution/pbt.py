import random
import copy
from typing import Tuple, List, Callable

from .evolution import EvolveEngine
from .member import MemberState, Generation
from .de.mutation import de_rand_1
from .utils.iterable import random_from_list

class ExploitAndExplore(EvolveEngine):
    """
    A general, modifiable implementation of PBTs exploitation and exploration method.
    """
    def __init__(self, exploit_factor = 0.2, explore_factors = (0.8, 1.2)):
        super().__init__()
        assert isinstance(exploit_factor, float) and 0.0 <= exploit_factor <= 1.0, f"Exploit factor must be of type {float} between 0.0 and 1.0."
        assert isinstance(explore_factors, (float, list, tuple)), f"Explore factors must be of type {float}, {tuple} or {list}."
        self.exploit_factor = exploit_factor
        self.explore_factors = explore_factors

    def on_member_spawn(self, member : MemberState, logger : Callable[[str], None]):
        """Called for each new member."""
        [hp.sample_uniform() for hp in member.parameters]

    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> MemberState:
        """ Exploit best peforming members and explores all search spaces with random perturbation. """
        for member in generation:
            candidate, is_exploited = self.exploit(member, generation, logger)
            if is_exploited:
                candidate = self.explore(candidate, logger)
            yield candidate

    def on_evaluation(self, candidate : MemberState, logger : Callable[[str], None]) -> MemberState:
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

class ExploitAndExploreWithDifferentialEvolution(ExploitAndExplore):
    """
    A general, modifiable implementation of PBTs exploitation and exploration method.
    """
    def __init__(self, exploit_factor = 0.2, F = 0.2, Cr = 0.8):
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

    def on_evaluation(self, candidates : Tuple[MemberState, MemberState], logger : Callable[[str], None]) -> MemberState:
        """Evaluates candidate, compares it to the base and returns the best performer."""
        return candidates

    def explore(self, member : MemberState, generation : Generation, logger : Callable[[str], None]) -> MemberState:
        if len(generation) < 3:
            raise ValueError("generation size must be at least 3 or higher.")
        candidate = member.copy()
        hp_dimension_size = len(member.parameters)
        x_r0, x_r1, x_r2 = random_from_list(generation, k=3, exclude=member)
        j_rand = random.randrange(0, hp_dimension_size)
        for j in range(hp_dimension_size):
            if random.uniform(0.0, 1.0) <= self.Cr or j == j_rand:
                candidate[j] = de_rand_1(F = self.F, x_r0 = x_r0[j], x_r1 = x_r1[j], x_r2 = x_r2[j])
            else:
                candidate[j] = member[j]
        return candidate