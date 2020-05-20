import os
import random
import argparse

import torch
import numpy as np
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

from main_helper import run

def validate_arguments(args):
    if (args.population_size < 1):
        raise ValueError("Population size must be at least 2.")
    if (args.batch_size < 0):
        raise ValueError("Batch size must be at least 1.")
    if (args.step_size < 0):
        raise ValueError("Step size must be at least 1.")
    if (args.eval_steps < 0):
        raise ValueError("Step size must satisfy 0 <= eval_steps < step_size.")
    if (args.step_size < 0):
        raise ValueError("Step size must be at least 1.")
    if any(device.startswith('cuda') for device in args.devices):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available on your machine.")
        if os.name == 'nt':
            raise NotImplementedError("PyTorch multiprocessing with CUDA is not supported on Windows.")
    return args

def import_user_arguments():
    # import user arguments
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--task", type=str, required=True, help="The name of the task, e.g. 'mnist_lenet5' or 'creditfraud'.")
    parser.add_argument("--evolver", type=str, required=True, help="Select which evolve algorithm to use.")
    parser.add_argument("--population_size", type=int, required=True, help="The number of members in the initial generation.")
    parser.add_argument("--batch_size", type=int, required=True, help="The number of batches in which the training set will be divided into.")
    parser.add_argument("--train_steps", type=int, required=True, help="Number of steps (or batches) to train each member in the generation.")
    parser.add_argument("--fitness_steps", type=int, required=True, default=8, help="Number of fitness steps (or batches) to measure each member in the generation. If 1 or higher, a random subset is sampled from the validation set.")
    parser.add_argument("--end_steps", type=int, default=None, help="Set number of steps as criterium for early stopping.")
    parser.add_argument("--end_time", type=int, default=None, help="Set how much time (in minutes) as a criterium for early stopping.")
    parser.add_argument("--end_score", type=float, default=None, help="Set the target score as criterium for early stopping.")
    parser.add_argument("--directory", type=str, default='checkpoints', help="Directory path to where the checkpoint database is to be located. Default: 'checkpoints/'.")
    parser.add_argument("--history", type=int, default=2, help="Number of generation states to save. Older generation states will be deleted.")
    parser.add_argument("--devices", type=str, default=['cpu'], nargs='*', help="Set processor device ('cpu' or 'cuda:0'). GPU is not supported on windows for PyTorch multiproccessing. Default: 'cpu'.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of training processes to perform at once.")
    parser.add_argument("--tensorboard", action='store_true', help="Wether to enable tensorboard 2.0 for real-time monitoring of the training process.")
    parser.add_argument("--logging", action='store_true', help="Wether to enable logging.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level.")
    args = parser.parse_args()
    validate_arguments(args)
    return args

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # set torch settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    args = import_user_arguments()
    validate_arguments(args)
    run(**vars(args))