import argparse
import os
import random
import numpy
import torch
from task import Mnist, EMnist, CreditCardFraud
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

def import_user_arguments():
    # import user arguments
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--population_size", type=int, default=10, help="The number of members in the population. Default: 5.")
    parser.add_argument("--batch_size", type=int, default=64, help="The number of batches in which the training set will be divided into.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to train each training process.")
    parser.add_argument("--task", type=str, default='creditfraud', help="Select tasks from 'mnist', 'creditfraud'.")
    parser.add_argument("--evolver", type=str, default='pbt', help="Select which evolve algorithm to use.")
    parser.add_argument("--database_path", type=str, default='checkpoints/creditfraud_pbt', help="Directory path to where the checkpoint database is to be located. Default: 'checkpoints/'.")
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
    if args.tensorboard:
        print(f"Launching tensorboard...")
        tensorboard_log_path = f"{database.path}/tensorboard"
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tensorboard_log_path])
        url = tb.launch()
        print(f"Tensoboard is launched and accessible at: {url}")
        tensorboard_writer = SummaryWriter(tensorboard_log_path)
    # prepare objective
    print(f"Importing task...")
    if args.task == "creditfraud":
        task = CreditCardFraud()
    if args.task == "mnist":
        task = Mnist()
    if args.task == "emnist":
        task = EMnist()
    # objective info
    print(f"Population size: {args.population_size}")
    print(f"Number of hyper-parameters: {len(task.hyper_parameters)}")
    print(f"Train data length: {len(task.train_data)}")
    print(f"Eval data length: {len(task.eval_data)}")
    print(f"Test data length: {len(task.test_data)}")
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
        load_in_memory=True)
    print(f"Creating evaluator...")
    EVALUATOR = Evaluator(
        model_class = task.model_class,
        test_data = task.eval_data,
        loss_functions=task.loss_functions,
        batch_size = args.batch_size,
        device = args.device,
        load_in_memory=True)
    print(f"Creating tester...")
    TESTER = Evaluator(
        model_class = task.model_class,
        test_data = task.test_data,
        loss_functions=task.loss_functions,
        batch_size = args.batch_size,
        device = args.device,
        load_in_memory=True)
    # define controller
    print(f"Creating evolver...")
    if args.evolver == 'pbt':
        EVOLVER = ExploitAndExplore(
            population_size = args.population_size,
            exploit_factor = 0.2,
            explore_factors = (0.8, 1.2),
            random_walk=False)
    if args.evolver == 'pbt_rw':
        EVOLVER = ExploitAndExplore(
            population_size = args.population_size,
            exploit_factor = 0.2,
            explore_factors = (0.8, 1.2),
            random_walk=True)
    if args.evolver == 'de':
        EVOLVER = DifferentialEvolution(
            population_size = args.population_size,
            F = 0.2,
            Cr = 0.8,
            constraint='clip')
    if args.evolver == 'de_r':
        EVOLVER = DifferentialEvolution(
            population_size = args.population_size,
            F = 0.2,
            Cr = 0.8,
            constraint='reflect')
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
        sensitivity=4)
    analyzer.create_hyper_parameter_single_plot_files(
        save_directory=database.create_folder("results/plots/hyper_parameters"),
        sensitivity=4)
    N_TEST_MEMBER_LIMIT = 50
    print(f"Testing the top {N_TEST_MEMBER_LIMIT} members on the test set of {len(task.test_data)} samples...")
    TESTED_CHECKPOINTS = analyzer.test(evaluator=TESTER, limit=N_TEST_MEMBER_LIMIT)
    for checkpoint in TESTED_CHECKPOINTS:
        database.update(checkpoint.id, checkpoint.steps, checkpoint, ignore_exception=True)
    BEST_CHECKPOINT = max(TESTED_CHECKPOINTS, key=lambda c: c.loss['test'][task.eval_metric])
    RESULT = f"Member {BEST_CHECKPOINT.id} performed best on epoch {BEST_CHECKPOINT.epochs} / step {BEST_CHECKPOINT.steps} with an {task.eval_metric} of {BEST_CHECKPOINT.loss['test'][task.eval_metric]:.4f}"
    database.create_file("results", "best_member.txt").write_text(RESULT)
    with database.create_file("results", "top_members.txt").open('a+') as f:
        for checkpoint in TESTED_CHECKPOINTS:
            f.write(str(checkpoint) + "\n")
    print(RESULT)
