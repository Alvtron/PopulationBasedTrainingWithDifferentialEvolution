import argparse
import os
import random
import numpy
import torch
from functools import partial
from task import Mnist, EMnist, FashionMnist, CreditCardFraud
from tensorboard import program
import torch
from torch.utils.tensorboard import SummaryWriter

from analyze import Analyzer
from controller import Controller
from database import Database
from evaluator import Evaluator
from evolution import DifferentialEvolution, ExploitAndExplore, LSHADE
from trainer import Trainer

# set random state for reproducibility
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.multiprocessing.set_sharing_strategy('file_descriptor')

TASK = 'mnist'
EVOLVER = 'pbt'
POPULATION_SIZE = 20
BATCH_SIZE = 64
STEP_SIZE = 100
END_STEPS = None # * STEP_SIZE
END_EPOCHS = None
END_NFE = POPULATION_SIZE * 100
END_SCORE = None
TEST_LIMIT = 100
HISTORY_LIMIT = 100
DIRECTORY = 'checkpoints'
DEVICE = 'cuda'
N_JOBS = 6

def import_arguments():
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--task", type=str, default=TASK, help="Select tasks from 'mnist', 'creditfraud'.")
    parser.add_argument("--evolver", type=str, default=EVOLVER, help="Select which evolve algorithm to use.")
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE, help="The number of members in the population. Default: 5.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="The number of batches in which the training set will be divided into.")
    parser.add_argument("--step_size", type=int, default=STEP_SIZE, help="Number of steps to train each training process.")
    parser.add_argument("--end_steps", type=int, default=END_STEPS, help="Perform early stopping after the specified number of steps.")
    parser.add_argument("--end_epochs", type=int, default=END_EPOCHS, help="Perform early stopping after the specified number of epochs.")
    parser.add_argument("--end_nfe", type=int, default=END_NFE, help="Perform early stopping after the specified number of fitness evaluations.")
    parser.add_argument("--end_score", type=int, default=END_SCORE, help="Perform early stopping when the specified score is met.")
    parser.add_argument("--test_limit", type=int, default=TEST_LIMIT, help="Number of top performing database entries to test with the testing set.")    
    parser.add_argument("--history_limit", type=int, default=HISTORY_LIMIT, help="Sets the number of network model- and optimizer states to keep stored in database.")
    parser.add_argument("--directory", type=str, default=DIRECTORY, help="The directory path to where the checkpoint database is to be located. Default: 'checkpoints/'.")
    parser.add_argument("--device", type=str, default=DEVICE, help="Sets the processor device ('cpu' or 'gpu' or 'cuda'). GPU is not supported on windows for PyTorch multiproccessing. Default: 'cpu'.")
    parser.add_argument("--n_jobs", type=int, default=N_JOBS, help="Sets the number of training processes. If n_jobs is less than 1 or higher than population size, the population size will be used instead.")
    parser.add_argument("--tensorboard", type=bool, default=True, help="Decides whether to enable tensorboard 2.0 for real-time monitoring of the training process.")
    parser.add_argument("--in_memory", type=bool, default=True, help="Decides whether the dataset will be loaded in memory.")    
    parser.add_argument("--detect_NaN", type=bool, default=True, help="Decides whether the controller will detect NaN loss values.")    
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity level")
    parser.add_argument("--logging", type=bool, default=True, help="Logging level")
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
        return CreditCardFraud()
    elif task_name == "mnist":
        return Mnist()
    elif task_name == "fashionmnist":
        return FashionMnist()
    elif task_name == "emnist":
        return EMnist()
    else:
        raise NotImplementedError(f"Your task request '{task_name}'' is not available.")

def create_evolver(evolver_name, population_size):
    if evolver_name == 'pbt':
        return ExploitAndExplore(
            exploit_factor = 0.2,
            explore_factors = (0.8, 1.2),
            random_walk=False)
    if evolver_name == 'pbt_rw':
        return ExploitAndExplore(
            exploit_factor = 0.2,
            explore_factors = (0.8, 1.2),
            random_walk=True)
    if evolver_name == 'de':
        return DifferentialEvolution(
            F = 0.2,
            Cr = 0.8,
            constraint='clip')
    if evolver_name == 'de_r':
        return DifferentialEvolution(
            F = 0.2,
            Cr = 0.8,
            constraint='reflect')
    if evolver_name == 'lshade':
        return LSHADE(
            N_INIT = population_size,
            MAX_NFE=args.end_nfe,
            r_arc=2.0,
            p=0.2,
            memory_size=5)
    else:
        raise NotImplementedError(f"Your evolver request '{evolver_name}'' is not available.")

def create_tensorboard(log_directory):
    tensorboard_log_path = f"{log_directory}/tensorboard"
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tensorboard_log_path])
    url = tb.launch()
    return SummaryWriter(tensorboard_log_path), url

if __name__ == "__main__":
    args = import_arguments()
    # prepare objective
    print(f"Importing task...")
    task = import_task(args.task)
    # prepare database
    print(f"Preparing database...")
    database = Database(f"{args.directory}/{task.name}/{args.evolver}")
    # prepare tensorboard writer
    if args.tensorboard:    
        print(f"Launching tensorboard...")
        tensorboard_writer, tensorboard_url = create_tensorboard(database.path)
        print(f"Tensoboard is launched and accessible at: {tensorboard_url}")
    # print and save objective info
    obj_info = [
        f"Task: {task.name}",
        f"Evolver: {args.evolver}",
        f"Database path: {database.path}",
        f"Population size: {args.population_size}",
        f"Hyper-parameters: {len(task.hyper_parameters)} {task.hyper_parameters.keys()}",
        f"Batch size: {args.batch_size}",
        f"Step size: {args.step_size}",
        f"End criterium - steps: {args.end_steps}",
        f"End criterium - epochs: {args.end_epochs}",
        f"End criterium - fitness evaluations: {args.end_nfe}",        
        f"End criterium - score: {args.end_score}",
        f"Loss-metric: {task.loss_metric}",
        f"Eval-metric: {task.eval_metric}",
        f"Loss functions: {[loss.name for loss in task.loss_functions.values()]}",
        f"Test limit: {args.test_limit}",
        f"History limit: {args.history_limit}",
        f"Detect NaN: {args.detect_NaN}",
        f"Training set length: {len(task.train_data)}",
        f"Evaluation set length: {len(task.eval_data)}",
        f"Testing set length: {len(task.test_data)}",
        f"Total dataset length: {len(task.train_data) + len(task.eval_data) + len(task.test_data)}",
        f"Load dataset in memory: {args.in_memory}",
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
        train_data = task.train_data,
        loss_functions = task.loss_functions,
        loss_metric = task.loss_metric,
        batch_size = args.batch_size,
        device = args.device,
        load_in_memory=args.in_memory)
    print(f"Creating evaluator...")
    EVALUATOR = Evaluator(
        model_class = task.model_class,
        test_data = task.eval_data,
        loss_functions=task.loss_functions,
        batch_size = args.batch_size,
        device = args.device,
        load_in_memory=args.in_memory)
    print(f"Creating tester...")
    TESTER = Evaluator(
        model_class = task.model_class,
        test_data = task.test_data,
        loss_functions=task.loss_functions,
        batch_size = args.batch_size,
        device = args.device,
        load_in_memory=args.in_memory)
    # define controller
    print(f"Creating evolver...")
    EVOLVER = create_evolver(args.evolver, args.population_size)
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
        evolve_frequency=args.step_size,
        end_criteria={'steps': args.end_steps, 'epochs': args.end_epochs, 'nfe': args.end_nfe, 'score': args.end_score},
        detect_NaN=args.detect_NaN,
        device=args.device,
        n_jobs=args.n_jobs,
        history_limit=args.history_limit,
        tensorboard_writer=tensorboard_writer,
        verbose=args.verbose,
        logging=args.logging)
    # run controller
    print(f"Starting controller...")
    controller.start()    # analyze results stored in database
    print("Analyzing population...")
    analyzer = Analyzer(database)
    print(f"Testing the top {args.test_limit} members on the test set of {len(task.test_data)} samples...")
    tested_checkpoints = analyzer.test(
        evaluator=TESTER,
        save_directory=database.create_file("results", "top_members.txt"),
        limit=args.test_limit,
        verbose=args.verbose)
    print("Updating database with results...")
    for checkpoint in tested_checkpoints:
        database.update(checkpoint.id, checkpoint.steps, checkpoint)
    print("Creating statistics...")
    analyzer.create_statistics(save_directory=database.create_folder("results/statistics"))
    print("Creating plot-files...")
    analyzer.create_plot_files(save_directory=database.create_folder("results/plots"))
    analyzer.create_hyper_parameter_plot_files(save_directory=database.create_folder("results/plots"), sensitivity=20)
    print("Program completed! You can now exit if needed.")