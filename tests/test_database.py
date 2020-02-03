import shutil
import itertools

import context
from pbt.database import ReadOnlyDatabase

print(f"Creating databse...")
database = ReadOnlyDatabase("checkpoints/mnist/pbt/20200121112716")
print(f"Database exist: {database.exists}")
print(f"Database consists of {len(database)} entries.")
print(f"Latest entries are:")
for entry in database.latest():
    print(entry)
print(f"Entry 200 on key 0: {database.entry(0, 200)}")
print(f"Id '0' exist: {0 in database}")
print(f"Id '50' exist: {50 in database}")
print(f"Length of entries on id 0: {len(list(database.entries(0)))}")
print(f"Print 10 first entries from iterator:")
for index, entry in enumerate(itertools.islice(database, 0, 10)):
    print(index, entry)
print(f"Print the entire database")
database.print()