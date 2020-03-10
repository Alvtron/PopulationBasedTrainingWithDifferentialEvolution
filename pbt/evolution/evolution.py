import random
import copy
from abc import ABC
from typing import Tuple, Callable, Generator

from ..member import MemberState, Generation

class EvolveEngine(ABC):
    """
    Base class for all evolvers.
    """

    def on_member_spawn(self, member : MemberState, logger : Callable[[str], None]):
        """Called for each new member."""
        pass

    def on_generation_start(self, generation : Generation, logger : Callable[[str], None]):
        """Called before each generation."""
        pass

    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> Generator[Tuple[MemberState, ...], None, None]:
        """Called for each member in generation. Returns one candidate or multiple candidates."""
        pass

    def on_evaluation(self, candidates : Tuple[MemberState, ...], logger : Callable[[str], None]) -> MemberState:
        """Returns the determined 'best' member between the candidates."""
        pass

    def on_generation_end(self, generation : Generation, logger : Callable[[str], None]):
        """Called at the end of each generation."""
        pass

class RandomSearch(EvolveEngine):
    def __init__(self):
        super().__init__()

    def on_member_spawn(self, member : MemberState, logger : Callable[[str], None]):
        """Called for each new member."""
        [hp.sample_uniform() for hp in member.parameters]

    def on_evolve(self, generation : Generation, logger : Callable[[str], None]) -> MemberState:
        """Simply returns the member. No mutation conducted."""
        for member in generation:
            yield member.copy()

    def on_evaluation(self, candidate : MemberState, logger : Callable[[str], None]) -> MemberState:
        """Simply returns the candidate. No evaluation conducted."""
        return candidate

class RandomWalk(EvolveEngine):
    def __init__(self, explore_factor = 0.2):
        super().__init__()
        self.explore_factor = explore_factor

    def on_member_spawn(self, member : MemberState, logger : Callable[[str], None]):
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

    def on_evaluation(self, candidate : MemberState, logger : Callable[[str], None]) -> MemberState:
        """Simply returns the candidate. No evaluation conducted."""
        return candidate



