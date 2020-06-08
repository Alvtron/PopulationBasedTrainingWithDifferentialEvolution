from typing import Sequence, Any

from pbt.database import Database
from pbt.member import Generation

class GarbageCollector:
    def __init__(self, database: Database, history_limit: int = None, verbose: bool = False):
        if not isinstance(database, Database):
            raise TypeError(f"the 'database' specified was of wrong type {type(database)}, expected {Database}.")
        if not isinstance(history_limit, int):
            raise TypeError(f"the 'history_limit' specified was of wrong type {type(history_limit)}, expected {int}.")
        if not isinstance(verbose, bool):
            raise TypeError(f"the 'verbose' specified was of wrong type {type(verbose)}, expected {bool}.")
        self.database = database
        self.history_limit = history_limit
        self.verbose = verbose
        self.__processed_identities = set()

    def _log(self, message: str) -> None:
        if not self.verbose:
            return
        print(f"GC: {message}")

    def collect(self, exclude: Sequence[Any] = None) -> None:
        if exclude is not None and not isinstance(exclude, (list, tuple)):
            raise TypeError(f"the 'exclude' specified was of wrong type {type(exclude)}, expected {list} or {tuple}.")
        if self.history_limit is None:
            return
        for uid, keys in self.database.identy_records().items():
            numeric_keys = sorted(int(key) for key in keys)
            for key in numeric_keys[:-self.history_limit]:
                if (uid, key) in self.__processed_identities:
                    # skipping the member as it has already been processed
                    continue
                member = self.database.entry(uid, key)
                if exclude is not None and member in exclude:
                    # skipping the member as it is requsted to be excluded
                    continue
                if not member.has_state():
                    # skipping the member as it has no state to delete
                    continue
                self._log(f"deleting the state from member {member.uid} at step {member.steps}...")
                member.delete_state()
                # updating database
                self.database.update(member.uid, member.steps, member)
                # save identiy to processed identities
                self.__processed_identities.add((uid, key))