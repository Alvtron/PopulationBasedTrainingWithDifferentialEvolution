from typing import Sequence

from pbt.database import Database
from pbt.member import Generation

class GarbageCollector:
    def __init__(self, database : Database, history_limit : int = None, verbose : int = 0):
        self.database = database
        self.history_limit = history_limit
        self.verbose = verbose
        self.__processed_identities = set()

    def _log(self, message : str) -> None:
        if self.verbose <= 0:
            return
        print(f"GC: {message}")

    def collect(self) -> None:
        if self.history_limit is None:
            return
        for uid, keys in self.database.identy_records().items():
            numeric_keys = sorted(int(key) for key in keys)
            for key in numeric_keys[:-self.history_limit]:
                if (uid, key) in self.__processed_identities:
                    # skipping the member as it has already been processed
                    continue
                member = self.database.entry(uid, key)
                if not member.has_state():
                    # skipping the member as it has no state to delete
                    continue
                self._log(f"deleting the state from member {member.id} at step {member.steps} with score {member.score():.4f}...")
                member.delete_state()
                # updating database
                self.database.update(member.id, member.steps, member)
                # save identiy to processed identities
                self.__processed_identities.add((uid, key))