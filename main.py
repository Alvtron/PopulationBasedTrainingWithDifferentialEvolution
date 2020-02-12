import os
import copy
import random
import time
import argparse
from functools import partial
from dataclasses import dataclass

import torch
import numpy as np
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

from pbt.task import mnist, creditfraud
from pbt.analyze import Analyzer
from pbt.controller import Controller
from pbt.database import Database
from pbt.evaluator import Evaluator
from pbt.evolution import RandomWalk, ExploitAndExplore, DifferentialEvolution, SHADE, LSHADE
from pbt.trainer import Trainer

# various settings for reproducibility
# set random state 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# set torch settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
# multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')

@dataclass
class Arguments():
    task : str
    evolver : str
    directory : str
    population_size : int = 10
    batch_size : int = 32
    step_size : int = 100
    end_nfe : int = None
    end_steps : int = 1000
    end_score : float = None
    history_limit : int = 2
    device : str = 'cpu'
    n_jobs : int = 1
    tensorboard : bool = False
    detect_NaN : bool = False
    verbose : int = 1
    logging : bool = False

def import_arguments():
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--task", type=str, help="Select tasks such as 'mnist', 'creditfraud'.")
    parser.add_argument("--evolver", type=str, help="Select which evolve algorithm to use.")
    parser.add_argument("--directory", type=str, help="The directory path to where the checkpoint database is to be located. Default: 'checkpoints/'.")
    parser.add_argument("--population_size", type=int, default=10, help="The number of members in the population.")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of batches in which the training set will be divided into.")
    parser.add_argument("--step_size", type=int, default=100, help="Number of steps to train each training process.")
    parser.add_argument("--end_nfe", type=int, default=None, help="Perform early stopping after the specified number of fitness evaluations.")
    parser.add_argument("--end_steps", type=int, default=100, help="Perform early stopping after the specified number of steps.")
    parser.add_argument("--end_score", type=int, default=None, help="Perform early stopping when the specified score is met.")
    parser.add_argument("--history_limit", type=int, default=2, help="Sets the number of network model- and optimizer states to keep stored in database.")
    parser.add_argument("--device", type=str, default='cpu', help="Sets the torch processor device ('cpu' or 'cuda'). GPU is not supported on windows for PyTorch multiproccessing. Default: 'cpu'.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Sets the number of training processes. If n_jobs is less than 1 or higher than population size, the population size will be used instead.")
    parser.add_argument("--tensorboard", type=bool, default=False, help="Decides whether to enable tensorboard 2.0 for real-time monitoring of the training process.")
    parser.add_argument("--detect_NaN", type=bool, default=True, help="Decides whether the controller will detect NaN loss values.")    
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level.")
    parser.add_argument("--logging", type=bool, default=True, help="Logging level.")
    args = parser.parse_args()
    args = validate_arguments(args)
    return args

def validate_arguments(args):
    if (args.population_size < 0):
        raise ValueError("Population size must be at least 1.")
    if (args.batch_size < 0):
        raise ValueError("Batch size must be at least 1.")
    if (args.step_size < 0):
        raise ValueError("Step size must be at least 1.")
    if (args.device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError("CUDA is not available on your machine.")
    if (args.device == 'cuda' and os.name == 'nt'):
        raise NotImplementedError("PyTorch multiprocessing with CUDA is not supported on Windows.")
    return args

def import_task(task_name : str):
    if task_name == "creditfraud":
        return creditfraud.CreditCardFraud()
    elif task_name == "mnist":
        return mnist.Mnist()
    elif task_name == "fashionmnist":
        return mnist.FashionMnist()
    elif task_name == "mnist_lenet5":
        return mnist.MnistKnowledgeSharing('lenet5')
    elif task_name == "mnist_mlp":
        return mnist.MnistKnowledgeSharing('mlp')
    elif task_name == "emnist_byclass":
        return mnist.EMnist("byclass")
    elif task_name == "emnist_bymerge":
        return mnist.EMnist("bymerge")
    elif task_name == "emnist_balanced":
        return mnist.EMnist("balanced")
    elif task_name == "emnist_letters":
        return mnist.EMnist("letters")
    elif task_name == "emnist_digits":
        return mnist.EMnist("digits")
    elif task_name == "emnist_mnist":
        return mnist.EMnist("mnist")
    else:
        raise NotImplementedError(f"Your requested task '{task_name}'' is not available.")

def create_evolver(evolver_name, population_size, end_nfe):
    if evolver_name == 'rw':
        return RandomWalk(explore_factor = 0.2)
    if evolver_name == 'pbt':
        return ExploitAndExplore( exploit_factor = 0.2, explore_factors = (0.9, 1.1))
    if evolver_name == 'de':
        return DifferentialEvolution( F = 0.2, Cr = 0.8)
    if evolver_name == 'shade':
        return SHADE( N_INIT = population_size, r_arc=2.0, p=0.2, memory_size=5)
    if evolver_name == 'lshade':
        return LSHADE( N_INIT = population_size, MAX_NFE=end_nfe, r_arc=2.0, p=0.2, memory_size=5)
    else:
        raise NotImplementedError(f"Your evolver request '{evolver_name}'' is not available.")

def create_tensorboard(log_directory):
    tensorboard_log_path = f"{log_directory}/tensorboard"
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tensorboard_log_path])
    url = tb.launch()
    return SummaryWriter(tensorboard_log_path), url

def run_task(args : Arguments):
    # prepare objective
    print(f"Importing task...")
    task = import_task(args.task)
    # prepare database
    print(f"Preparing database...")
    database = Database(
        directory_path=f"{args.directory}/{args.task}/{args.population_size}/{args.evolver}",
        read_function=torch.load, write_function=torch.save)
    # prepare tensorboard writer
    tensorboard_writer = None
    if args.tensorboard:
        print(f"Launching tensorboard...")
        tensorboard_writer, tensorboard_url = create_tensorboard(database.path)
        print(f"Tensoboard is launched and accessible at: {tensorboard_url}")
    # print and save objective info
    obj_info = [
        f"Task: {args.task}",
        f"Evolver: {args.evolver}",
        f"Database path: {database.path}",
        f"Population size: {args.population_size}",
        f"Hyper-parameters: {len(task.hyper_parameters)} {task.hyper_parameters.keys()}",
        f"Batch size: {args.batch_size}",
        f"Step size: {args.step_size}",
        f"End criterium - fitness evaluations: {args.end_nfe}",        
        f"End criterium - steps: {args.end_steps}",
        f"End criterium - score: {args.end_score}",
        f"Loss-metric: {task.loss_metric}",
        f"Eval-metric: {task.eval_metric}",
        f"Loss functions: {[loss.name for loss in task.loss_functions.values()]}",
        f"History limit: {args.history_limit}",
        f"Detect NaN: {args.detect_NaN}",
        f"Training set length: {len(task.datasets.train)}",
        f"Evaluation set length: {len(task.datasets.eval)}",
        f"Testing set length: {len(task.datasets.test)}",
        f"Total dataset length: {len(task.datasets.train) + len(task.datasets.eval) + len(task.datasets.test)}",
        f"Verbosity: {args.verbose}",
        f"Logging: {args.logging}",
        f"Device: {args.device}",
        f"Number of processes: {args.n_jobs}"]
    if args.device == "cuda":
        obj_info.append(f"Number of GPUs: {torch.cuda.device_count()}")
    obj_info = "\n".join(obj_info)
    print("\n", obj_info, "\n")
    database.create_file(tag="info", file_name="information.txt").write_text(obj_info)
    # create trainer, evaluator and tester
    print(f"Creating trainer...")
    TRAINER = Trainer(
        model_class = task.model_class,
        optimizer_class = task.optimizer_class,
        train_data = task.datasets.train,
        loss_functions = task.loss_functions,
        loss_metric = task.loss_metric,
        batch_size = args.batch_size,
        device = args.device)
    print(f"Creating evaluator...")
    EVALUATOR = Evaluator(
        model_class = task.model_class,
        test_data = task.datasets.eval,
        loss_functions=task.loss_functions,
        batch_size = args.batch_size,
        device = args.device)
    print(f"Creating tester...")
    TESTER = Evaluator(
        model_class = task.model_class,
        test_data = task.datasets.test,
        loss_functions=task.loss_functions,
        batch_size = args.batch_size,
        device = args.device)
    # define controller
    print(f"Creating evolver...")
    EVOLVER = create_evolver(args.evolver, args.population_size, args.end_nfe)
    # create controller
    print(f"Creating controller...")
    controller = Controller(
        context = torch.multiprocessing.get_context('spawn'),
        population_size=args.population_size,
        hyper_parameters=task.hyper_parameters,
        trainer=TRAINER,
        evaluator=EVALUATOR,
        evolver=EVOLVER,
        loss_metric=task.loss_metric,
        eval_metric=task.eval_metric,
        loss_functions=task.loss_functions,
        database=database,
        step_size=args.step_size,
        eval_steps=1,
        end_criteria={'nfe': args.end_nfe, 'steps': args.end_steps, 'score': args.end_score},
        detect_NaN=args.detect_NaN,
        device=args.device,
        n_jobs=args.n_jobs,
        history_limit=args.history_limit,
        tensorboard_writer=tensorboard_writer,
        verbose=args.verbose,
        logging=args.logging)
    # run controller
    print(f"Starting controller...")
    controller.start() 
    # analyze results stored in database
    print("Analyzing population...")
    analyzer = Analyzer(database)
    checkpoint = analyzer.test(
        evaluator=TESTER,
        save_directory=database.create_file("results", "top_members.txt"),
        verbose=args.verbose)
    print("Updating database with results...")
    database.update(checkpoint.id, checkpoint.steps, checkpoint)
    print("Creating statistics...")
    analyzer.create_statistics(save_directory=database.create_folder("results/statistics"))
    print("Creating plot-files...")
    analyzer.create_plot_files(save_directory=database.create_folder("results/plots"))
    analyzer.create_hyper_parameter_plot_files(save_directory=database.create_folder("results/plots"), sensitivity=20)
    print("Program completed! You can now exit if needed.")

"""
        CONFIGURATIONS
"""

def test_mnist(population_size = 30, evolver='lshade'):
    args = Arguments( task = 'mnist', evolver = evolver, population_size = population_size, batch_size = 64,
        step_size = 250, end_nfe = population_size * 40, end_steps = None, end_score = None, history_limit = 2,
        directory = 'checkpoints', device = 'cuda', n_jobs = 5, tensorboard = True, detect_NaN = True, verbose = 1, logging = True)
    run_task(args)
    time.sleep(10)

def test_like_knowledge_sharing_lenet5(population_size = 30, evolver='lshade'):
    args = Arguments( task = 'mnist_lenet5', evolver = evolver, population_size = population_size, batch_size = 64,
        step_size = 250, end_nfe = population_size * 40, end_steps = None, end_score = None, history_limit = 2,
        directory = 'checkpoints', device = 'cuda', n_jobs = 6, tensorboard = True, detect_NaN = True, verbose = 1, logging = True)
    run_task(args)
    time.sleep(10)

def test_like_knowledge_sharing_mlp(population_size = 30, evolver='lshade'):
    args = Arguments( task = 'mnist_mlp', evolver = evolver, population_size = population_size, batch_size = 64,
        step_size = 250, end_nfe = population_size * 40, end_steps = None, end_score = None, history_limit = 2,
        directory = 'checkpoints', device = 'cuda', n_jobs = 6, tensorboard = True, detect_NaN = True, verbose = 1, logging = True)
    run_task(args)
    time.sleep(10)

"""
        PROGRAM STARTS HERE
"""

if __name__ == "__main__":
    #args = import_arguments()
    #test_mnist(population_size = 30, evolver='lshade')
    #test_mnist(population_size = 30, evolver='shade')
    #test_mnist(population_size = 30, evolver='de')
    #test_mnist(population_size = 30, evolver='pbt')
    test_like_knowledge_sharing_lenet5(population_size = 30, evolver='lshade')
    test_like_knowledge_sharing_lenet5(population_size = 30, evolver='pbt')
    test_like_knowledge_sharing_lenet5(population_size = 30, evolver='shade')
    test_like_knowledge_sharing_lenet5(population_size = 30, evolver='de')
    test_like_knowledge_sharing_mlp(population_size = 30, evolver='lshade')
    test_like_knowledge_sharing_mlp(population_size = 30, evolver='pbt')
    test_like_knowledge_sharing_mlp(population_size = 30, evolver='shade')
    test_like_knowledge_sharing_mlp(population_size = 30, evolver='de')

    # b = 64
    # x = 250
    # tren x steg frem
    # evaluer
    # mutasjon
    # tren mutasjon 1 steg frem
    # evaluer mutasjon
    # repeat

    # [b0, b1, b2, b3, b4, ..., bn]