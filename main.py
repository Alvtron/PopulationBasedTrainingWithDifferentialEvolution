import argparse
import os
import random
import numpy
import torch
from task import Mnist, EMnist, FashionMnist, CreditCardFraud
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

from analyze import Analyzer
from controller import Controller
from database import SharedDatabase
from evaluator import Evaluator
from evolution import DifferentialEvolution, ExploitAndExplore
from trainer import Trainer

# reproducibility
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def import_arguments():
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--task", type=str, default='fashionmnist', help="Select tasks from 'mnist', 'creditfraud'.")
    parser.add_argument("--evolver", type=str, default='de', help="Select which evolve algorithm to use.")
    parser.add_argument("--population_size", type=int, default=10, help="The number of members in the population. Default: 5.")
    parser.add_argument("--batch_size", type=int, default=64, help="The number of batches in which the training set will be divided into.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to train each training process.")
    parser.add_argument("--directory", type=str, default='checkpoints', help="The directory path to where the checkpoint database is to be located. Default: 'checkpoints/'.")
    parser.add_argument("--device", type=str, default='cpu', help="Sets the processor device ('cpu' or 'gpu' or 'cuda'). GPU is not supported on windows for PyTorch multiproccessing. Default: 'cpu'.")
    parser.add_argument("--tensorboard", type=bool, default=True, help="Decides whether to enable tensorboard 2.0 for real-time monitoring of the training process.")
    parser.add_argument("--in_memory", type=bool, default=True, help="Decides whether the dataset will be loaded in memory.")    
    parser.add_argument("--detect_NaN", type=bool, default=True, help="Decides whether the controller will detect NaN loss values.")    
    parser.add_argument("--verbose", type=bool, default=True, help="Verbosity level")
    parser.add_argument("--logging", type=bool, default=True, help="Logging level")
    args = parser.parse_args()
    args = validate_arguments(args)
    return args

def validate_arguments(args):
    if (args.population_size < 0):
        raise ValueError("Population size must be at least 1.")
    if (args.batch_size < 0):
        raise ValueError("Batch size must be at least 1.")
    if (args.steps < 0):
        raise ValueError("Step size must be at least 1.")
    if (args.steps < 0):
        raise ValueError("Step size must be at least 1.")
    if (args.device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError("CUDA is not available on your machine.")
    if (args.device == 'cuda' and os.name == 'nt'):
        raise NotImplementedError("Pytorch with CUDA is not supported on Windows.")
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
            population_size = population_size,
            exploit_factor = 0.2,
            explore_factors = (0.8, 1.2),
            random_walk=False)
    if evolver_name == 'pbt_rw':
        return ExploitAndExplore(
            population_size = population_size,
            exploit_factor = 0.2,
            explore_factors = (0.8, 1.2),
            random_walk=True)
    if evolver_name == 'de':
        return DifferentialEvolution(
            population_size = population_size,
            F = 0.2,
            Cr = 0.8,
            constraint='clip')
    if evolver_name == 'de_r':
        return DifferentialEvolution(
            population_size = population_size,
            F = 0.2,
            Cr = 0.8,
            constraint='reflect')
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
    database = SharedDatabase(
        context=torch.multiprocessing.get_context('spawn'),
        directory_path = f"{args.directory}/{task.name}/{args.evolver}",
        read_function=torch.load,
        write_function=torch.save)
    # prepare tensorboard writer
    if args.tensorboard:
        print(f"Launching tensorboard...")
        tensorboard_writer, tensorboard_url = create_tensorboard(database.path)
        print(f"Tensoboard is launched and accessible at: {tensorboard_url}")
    # print and save objective info
    obj_info = "\n".join([
        f"Task: {task.name}",
        f"Evolver: {args.evolver}",
        f"Database path: {database.path}",
        f"Population size: {args.population_size}",
        f"Hyper-parameters: {len(task.hyper_parameters)} {task.hyper_parameters.names()}",
        f"Batch size: {args.batch_size}",
        f"Step size: {args.steps}",
        f"Training set length: {len(task.train_data)}",
        f"Evaluation set length: {len(task.eval_data)}",
        f"Testing set length: {len(task.test_data)}",
        f"Total dataset length: {len(task.train_data) + len(task.eval_data) + len(task.test_data)}",
        f"Loss-metric: {task.loss_metric}",
        f"Eval-metric: {task.eval_metric}",
        f"Loss functions: {[loss.name for loss in task.loss_functions.values()]}",
        f"Load dataset in memory: {args.in_memory}",
        f"Detect NaN: {args.detect_NaN}",
        f"Device: {args.device}",
        f"Verbose: {args.verbose}",
        f"Logging: {args.logging}"])
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
        population_size=args.population_size,
        hyper_parameters=task.hyper_parameters,
        trainer=TRAINER,
        evaluator=EVALUATOR,
        tester=TESTER,
        evolver=EVOLVER,
        loss_metric=task.loss_metric,
        eval_metric=task.eval_metric,
        loss_functions=task.loss_functions,
        database=database,
        step_size=args.steps,
        evolve_frequency=args.steps,
        end_criteria={'steps': args.steps * 100, 'score': 100.0},
        detect_NaN=args.detect_NaN,
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
    analyzer.create_statistics(save_directory=database.create_folder("results/statistics"))
    print("Creating plot-files...")
    analyzer.create_plot_files(save_directory=database.create_folder("results/plots"))
    analyzer.create_hyper_parameter_multi_plot_files(
        save_directory=database.create_folder("results/plots/hyper_parameters"),
        sensitivity=4)
    analyzer.create_hyper_parameter_single_plot_files(
        save_directory=database.create_folder("results/plots/hyper_parameters"),
        sensitivity=4)
    N_TEST_MEMBER_LIMIT = 50
    print(f"Testing the top {N_TEST_MEMBER_LIMIT} members on the test set of {len(task.test_data)} samples...")
    TESTED_CHECKPOINTS = analyzer.test(evaluator=TESTER, limit=N_TEST_MEMBER_LIMIT)
    for checkpoint in TESTED_CHECKPOINTS:
        database.update(checkpoint.id, checkpoint.steps, checkpoint, ignore_exception=True)
    if task.loss_functions[task.eval_metric].minimize:  
        BEST_CHECKPOINT = min(TESTED_CHECKPOINTS, key=lambda c: c.loss['test'][task.eval_metric])
    else:
        BEST_CHECKPOINT = max(TESTED_CHECKPOINTS, key=lambda c: c.loss['test'][task.eval_metric])
    RESULT = f"Best checkpoint: {BEST_CHECKPOINT}"
    with database.create_file("results", "top_members.txt").open('a+') as f:
        f.write(f"{RESULT}\n\n")
        for checkpoint in TESTED_CHECKPOINTS:
            f.write(str(checkpoint) + "\n")
    print(RESULT)
