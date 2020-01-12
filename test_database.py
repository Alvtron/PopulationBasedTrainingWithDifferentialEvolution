import torch
import shutil
import itertools
from database import ReadOnlyDatabase

print(f"Creating databse...")
database = ReadOnlyDatabase(
    database_path="checkpoints/mnist_de_clip/20200110224544_98.1986",
    read_function=torch.load)
print(f"Databse exist: {database.exists}")
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