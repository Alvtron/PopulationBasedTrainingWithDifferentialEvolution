from typing import Sequence

from pbt.database import Database
from pbt.member import Generation

class GarbageCollector:
    def __init__(self, database : Database, history_limit : int = None, verbose : int = 0):
        self.database = database
        self.history_limit = history_limit
        self.verbose = verbose

    def _log(self, message : str) -> None:
        if verbose <= 0:
            return
        print(f"GC: {message}")

    def collect(self, generations : Sequence[Generation]) -> None:
        if self.history_limit is None:
            return
        if len(generations) < self.history_limit + 1:
            return
        for member in generations[-(self.history_limit + 1)]:
            if not member.has_state():
                # skipping the member as it has no state to delete
                continue
            self._log(f"deleting the state from member {member.id} at step {member.steps} with score {member.score():.4f}...")
            member.delete_state()
            # updating database
            self.database.update(member.id, member.steps, member)