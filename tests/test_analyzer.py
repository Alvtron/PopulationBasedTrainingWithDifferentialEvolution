import unittest
import shutil
import random
from functools import partial
from pathlib import Path

import torch
import numpy as np

from pbt.analyze import Analyzer 
from pbt.database import ReadOnlyDatabase
from pbt.loss import CategoricalCrossEntropy, Accuracy, F1
from pbt.evaluator import Evaluator
from pbt.task.mnist import Mnist
from pbt.task.fashionmnist import FashionMnist

class TestAnalyzer(unittest.TestCase):
    def test_analyzer(self):
        batch_size = 64
        device = 'cuda'
        data_path = 'tests/checkpoint/lshade_20200422231243'
        task = FashionMnist(model='lenet5')
        tester = Evaluator(
            model_class=task.model_class,
            test_data=task.datasets.test,
            batch_size=batch_size,
            loss_functions=task.loss_functions)
        result_folder = Path("tests/analyzer_output")
        statistics_save_directory = Path(result_folder, "statistics")
        plot_save_directory = Path(result_folder, "plots")
        shutil.rmtree(result_folder, ignore_errors=True)
        statistics_save_directory.mkdir(exist_ok=True, parents=True)
        plot_save_directory.mkdir(exist_ok=True, parents=True)
        database = ReadOnlyDatabase(data_path, read_function=partial(torch.load, map_location=device))
        analyzer = Analyzer(database, verbose = False)
        analyzer.test(evaluator=tester, save_directory = Path(result_folder, "best_member.txt"), device = device)
        analyzer.create_statistics(save_directory=statistics_save_directory)
        analyzer.create_loss_plot_files(save_directory=plot_save_directory)
        analyzer.create_time_plot_files(save_directory=plot_save_directory)
        analyzer.create_hyper_parameter_plot_files(save_directory=plot_save_directory)
