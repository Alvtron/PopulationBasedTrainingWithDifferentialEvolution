import os
import sys
import argparse
import math
import torch
import torchvision
import torch.utils.data
import pandas
import numpy
import random
import sklearn.preprocessing
import sklearn.model_selection
import torchvision.transforms as transforms
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from model import MnistNet, FraudNet
from database import SharedDatabase
from hyperparameters import Hyperparameter, Hyperparameters
from controller import Controller
from evaluator import Evaluator
from trainer import Trainer
from evolution import ExploitAndExplore, DifferentialEvolution, ParticleSwarm
from analyze import Analyzer

# prepare database
print(f"Preparing database...")
mp = torch.multiprocessing.get_context('spawn')
manager = mp.Manager()
shared_memory_dict = manager.dict()
database = SharedDatabase(
    directory_path = "checkpoints/mnist/20200107091623",
    shared_memory_dict = shared_memory_dict)
# prepare test set
test_data = MNIST(
    './data',
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]))
# prepare evaluator
tester = Evaluator(
    model_class = MnistNet,
    batch_size = 32,
    test_data = test_data,
    device = 'cpu',
    verbose = False)
# create analyzer
analyzer = Analyzer(database, tester)
print("Database entries:")
database.print()
print("Analyzing population...")
analyzer.create_plot_files(
    n_hyper_parameters=9,
    min_score=0,
    max_score=100,
    annotate=False,
    transparent=False)
all_checkpoints = analyzer.test(limit=50)
if all_checkpoints:
    best_checkpoint = max(all_checkpoints, key=lambda c: c.test_score)
    print("Results...")
    result = f"Member {best_checkpoint.id} performed best on epoch {best_checkpoint.epochs} / step {best_checkpoint.steps} with an accuracy of {best_checkpoint.test_score:.4f}%"
    database.append_to_file("results.txt", result)
    print(result)