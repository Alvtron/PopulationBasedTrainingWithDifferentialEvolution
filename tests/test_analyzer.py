import shutil
import random
from functools import partial
from pathlib import Path

import torch
import numpy as np

import context
from pbt.analyze import Analyzer 
from pbt.database import ReadOnlyDatabase
from pbt.loss import CategoricalCrossEntropy, Accuracy, F1
from pbt.evaluator import Evaluator
from pbt.task.mnist import Mnist, FashionMnist

# various settings for reproducibility
# set random state 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# set torch settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

batch_size = 64
device = 'cuda'

loss_functions = {
    'cce': CategoricalCrossEntropy(),
    'acc': Accuracy()
}

task = FashionMnist()

tester = Evaluator(
    model_class=task.model_class,
    test_data=task.datasets.test,
    batch_size=batch_size,
    loss_functions=task.loss_functions)

result_folder = Path("test_output/analyzer")
statistics_save_directory = Path(result_folder, "statistics")
plot_save_directory = Path(result_folder, "plots")
shutil.rmtree(result_folder, ignore_errors=True)
statistics_save_directory.mkdir(exist_ok=True, parents=True)
plot_save_directory.mkdir(exist_ok=True, parents=True)
database = ReadOnlyDatabase("checkpoints/fashionmnist/p60_steps250_batch64_nfe2400/lshade/20200220100311", read_function=partial(torch.load, map_location=device))
analyzer = Analyzer(database, verbose = True)
print(f"Database consists of {len(database)} entries.")
print("Testing best member...")
analyzer.test_generations(evaluator=tester, save_directory = Path(result_folder, "best_member.txt"), device = device)
print("Creating statistics...")
analyzer.create_statistics(save_directory=statistics_save_directory)
print("create_plot_files...")
analyzer.create_plot_files(save_directory=plot_save_directory)
print("create_hyper_parameter_multi_plot_files...")
analyzer.create_hyper_parameter_plot_files(save_directory=plot_save_directory, sensitivity=20)
print("Analyze completed.")
