import torch
import shutil
from pathlib import Path

from analyze import Analyzer
from database import ReadOnlyDatabase

statistics_save_directory = Path("test/analyzer/statistics")
plot_save_directory = Path("test/analyzer/plots")
shutil.rmtree("test/analyzer", ignore_errors=True)
statistics_save_directory.mkdir(exist_ok=True, parents=True)
plot_save_directory.mkdir(exist_ok=True, parents=True)
database = ReadOnlyDatabase("checkpoints/mnist/20/lshade/20200130144212")
analyzer = Analyzer(database)
print(f"Database consists of {len(database)} entries.")
print("Creating statistics...")
analyzer.create_statistics(save_directory=statistics_save_directory)
print("create_plot_files...", end =" ")
analyzer.create_plot_files(save_directory=plot_save_directory)
print("done!")
print("create_hyper_parameter_multi_plot_files...")
analyzer.create_hyper_parameter_plot_files(save_directory=plot_save_directory, sensitivity=20)
print("Analyze completed.")
