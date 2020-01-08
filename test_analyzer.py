import torch
import shutil
from pathlib import Path

from analyze import Analyzer
from database import ReadOnlyDatabase

save_directory = Path("test/analyzer")
shutil.rmtree(save_directory, ignore_errors=True)
save_directory.mkdir(exist_ok=True, parents=True)
database = ReadOnlyDatabase(
    database_path="checkpoints/mnist/20200108134018",
    read_function=torch.load)
analyzer = Analyzer(database)
print(f"Database consists of {len(database.to_list())} entries.")
print("Creating statistics...")
analyzer.create_statistics(save_directory=save_directory, verbose=False)
print("Creating plot-files...")
analyzer.create_plot_files(save_directory=save_directory)
analyzer.create_hyper_parameter_plot_files(
    save_directory=save_directory,
    min_score=0,
    max_score=100,
    annotate=False,
    sensitivity=4)
