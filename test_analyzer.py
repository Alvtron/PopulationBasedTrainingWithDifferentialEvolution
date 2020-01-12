import torch
import shutil
from pathlib import Path

from analyze import Analyzer
from database import ReadOnlyDatabase

statistics_save_directory = Path("test/analyzer/statistics")
plot_save_directory = Path("test/analyzer/plots")
hp_plot_save_directory = Path("test/analyzer/plots/hp_plots")
shutil.rmtree("test/analyzer", ignore_errors=True)
statistics_save_directory.mkdir(exist_ok=True, parents=True)
plot_save_directory.mkdir(exist_ok=True, parents=True)
hp_plot_save_directory.mkdir(exist_ok=True, parents=True)
database = ReadOnlyDatabase(
    database_path="checkpoints/mnist_de_clip/20200110184314",
    read_function=torch.load)
analyzer = Analyzer(database)
print(f"Database consists of {len(database)} entries.")
print("Creating statistics...")
analyzer.create_statistics(save_directory=statistics_save_directory, verbose=False)
print("create_plot_files...", end =" ")
analyzer.create_plot_files(save_directory=plot_save_directory)
print("done!")
print("create_hyper_parameter_single_plot_files...", end =" ")
analyzer.create_hyper_parameter_single_plot_files(
    save_directory=hp_plot_save_directory,
    min_score=0,
    max_score=100,
    sensitivity=4)
print("done!")
print("create_hyper_parameter_multi_plot_files...", end =" ")
analyzer.create_hyper_parameter_multi_plot_files(
    save_directory=hp_plot_save_directory,
    min_score=0,
    max_score=100,
    sensitivity=4)
print("done!")
print("Analyze completed.")
