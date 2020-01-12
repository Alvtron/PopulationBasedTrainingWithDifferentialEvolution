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
import matplotlib.pyplot as plt
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from setup import setup_mnist, setup_emnist, setup_fraud
from database import SharedDatabase
from hyperparameters import Hyperparameter, Hyperparameters
from controller import Controller
from evaluator import Evaluator
from trainer import Trainer
from evolution import ExploitAndExplore, DifferentialEvolution, ParticleSwarm
from analyze import Analyzer

# reproducibility
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def import_user_arguments():
    # import user arguments
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--population_size", type=int, default=10, help="The number of members in the population. Default: 5.")
    parser.add_argument("--batch_size", type=int, default=64, help="The number of batches in which the training set will be divided into.")
    parser.add_argument("--task", type=str, default='mnist', help="Select tasks from 'mnist', 'fraud'.")
    parser.add_argument("--evolver", type=str, default='pbt', help="Select which evolve algorithm to use.")
    parser.add_argument("--database_path", type=str, default='checkpoints/mnistnet2_pbt_random_walk', help="Directory path to where the checkpoint database is to be located. Default: 'checkpoints/'.")
    parser.add_argument("--device", type=str, default='cpu', help="Set processor device ('cpu' or 'gpu' or 'cuda'). GPU is not supported on windows for PyTorch multiproccessing. Default: 'cpu'.")
    parser.add_argument("--tensorboard", type=bool, default=True, help="Wether to enable tensorboard 2.0 for real-time monitoring of the training process.")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbosity level")
    parser.add_argument("--logging", type=bool, default=True, help="Logging level")
    args = parser.parse_args()
    # argument error handling
    if (args.device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError("CUDA is not available on your machine.")
    if (args.device == 'cuda' and os.name == 'nt'):
        raise NotImplementedError("Pytorch with CUDA is not supported on Windows.")
    if (args.population_size < 1):
        raise ValueError("Population size must be at least 1.")
    if (args.batch_size < 1):
        raise ValueError("Batch size must be at least 1.")
    return args

if __name__ == "__main__":
    print(f"Importing user arguments...")
    args = import_user_arguments()
    # prepare database
    print(f"Preparing database...")
    database = SharedDatabase(
        context=torch.multiprocessing.get_context('spawn'),
        directory_path = args.database_path,
        read_function=torch.load,
        write_function=torch.save)
    print(f"The shared database is available at: {database.path}")
    # prepare tensorboard writer
    tensorboard_writer = None
    if args.tensorboard:
        print(f"Launching tensorboard...")
        tensorboard_log_path = f"{database.path}/tensorboard"
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tensorboard_log_path])
        url = tb.launch()
        print(f"Tensoboard is launched and accessible at: {url}")
        tensorboard_writer = SummaryWriter(tensorboard_log_path)
    # prepare objective
    print(f"Preparing model and datasets...")
    if args.task == "fraud":
        model_class, optimizer_class, loss_metric, eval_metric, eval_metrics, train_data, eval_data, test_data, hyper_parameters = setup_fraud()
    if args.task == "mnist":
        model_class, optimizer_class, loss_metric, eval_metric, eval_metrics, train_data, eval_data, test_data, hyper_parameters = setup_mnist()
    if args.task == "emnist":
        model_class, optimizer_class, loss_metric, eval_metric, eval_metrics, train_data, eval_data, test_data, hyper_parameters = setup_emnist()
    # objective info
    print(f"Population size: {args.population_size}")
    print(f"Number of hyper-parameters: {len(hyper_parameters)}")
    print(f"Train data length: {len(train_data)}")
    print(f"Eval data length: {len(eval_data)}")
    print(f"Test data length: {len(test_data)}")
    # create trainer, evaluator and tester
    print(f"Creating trainer...")
    trainer = Trainer(
        model_class = model_class,
        optimizer_class = optimizer_class,
        loss_metric = loss_metric,
        eval_metrics = eval_metrics,
        batch_size = args.batch_size,
        train_data = train_data,
        device = args.device,
        load_in_memory=True,
        verbose = False)
    print(f"Creating evaluator...")
    evaluator = Evaluator(
        model_class = model_class,
        eval_metrics = eval_metrics,
        batch_size = args.batch_size,
        test_data = eval_data,
        device = args.device,
        load_in_memory=True,
        verbose = False)
    print(f"Creating tester...")
    tester = Evaluator(
        model_class = model_class,
        eval_metrics = eval_metrics,
        batch_size = args.batch_size,
        test_data = test_data,
        device = args.device,
        load_in_memory=True,
        verbose = False)
    # define controller
    print(f"Creating evolver...")
    steps = 100
    end_criteria = {'steps': steps * 100, 'score': 100.0} #400*10**3
    if args.evolver == 'pbt':
        evolver = ExploitAndExplore(N = args.population_size, exploit_factor = 0.2, explore_factors = (0.8, 1.2), random_walk=True)
    if args.evolver == 'de':
        evolver = DifferentialEvolution(N = args.population_size, F = 0.2, Cr = 0.8, constraint='clip')
    # create controller
    print(f"Creating controller...")
    controller = Controller(
        population_size=args.population_size,
        hyper_parameters=hyper_parameters,
        trainer=trainer,
        evaluator=evaluator,
        tester=tester,
        evolver=evolver,
        loss_metric=loss_metric,
        eval_metric=eval_metric,
        database=database,
        step_size=steps,
        evolve_frequency=steps,
        end_criteria=end_criteria,
        detect_NaN=False,
        device=args.device,
        tensorboard_writer=tensorboard_writer,
        verbose=args.verbose,
        logging=args.logging)
    # run controller
    print(f"Starting controller...")
    controller.start()
    # analyze results stored in database
    print("Analyzing population...")
    analyzer = Analyzer(database)
    print("Creating statistics...")
    analyzer.create_statistics(save_directory=database.create_folder("results/statistics"), verbose=False)
    print("Creating plot-files...")
    analyzer.create_plot_files(save_directory=database.create_folder("results/plots"))
    analyzer.create_hyper_parameter_multi_plot_files(
        save_directory=database.create_folder("results/plots/hyper_parameters"),
        min_score=0,
        max_score=100,
        sensitivity=4)
    analyzer.create_hyper_parameter_single_plot_files(
        save_directory=database.create_folder("results/plots/hyper_parameters"),
        min_score=0,
        max_score=100,
        sensitivity=4)
    n_members_to_be_tested = 50
    print(f"Testing the top {n_members_to_be_tested} members on the test set of {len(test_data)} samples...")
    all_checkpoints = analyzer.test(evaluator=tester, limit=n_members_to_be_tested)
    for checkpoint in all_checkpoints:
        database.update(checkpoint.id, checkpoint.steps, checkpoint)
    best_checkpoint = max(all_checkpoints, key=lambda c: c.loss['test'][eval_metric])
    result = f"Member {best_checkpoint.id} performed best on epoch {best_checkpoint.epochs} / step {best_checkpoint.steps} with an {eval_metric} of {best_checkpoint.loss['test'][eval_metric]:.4f}"
    database.create_file("results", "best_member.txt").write_text(result)
    with database.create_file("results", "top_members.txt").open('a+') as f:
        for checkpoint in all_checkpoints:
            f.write(str(checkpoint) + "\n")
    print(result)