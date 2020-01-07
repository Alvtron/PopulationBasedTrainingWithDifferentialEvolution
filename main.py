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
from torchvision.datasets import MNIST
from model import MnistNet, FraudNet
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

def split_dataset(dataset, fraction):
        assert 0.0 <= fraction <= 1.0, f"The provided fraction must be between 0.0 and 1.0!"
        dataset_length = len(dataset)
        first_set_length = math.floor(fraction * dataset_length)
        second_set_length = dataset_length - first_set_length
        first_set, second_set = torch.utils.data.random_split(dataset, (first_set_length, second_set_length))
        return first_set, second_set

def setup_mnist():
    model_class = MnistNet
    optimizer_class = torch.optim.SGD
    loss_function = torch.nn.CrossEntropyLoss()
    # prepare training and testing data
    train_data_path = test_data_path = './data'
    train_data = MNIST(
        train_data_path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]))
    test_data = MNIST(
        test_data_path,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]))
    # split training set into training set and validation set
    train_data, eval_data = split_dataset(train_data, 0.90)
    # define hyper-parameter search space
    hyper_parameters = Hyperparameters(
        general_params = None,
        model_params = {
            'dropout_rate_1': Hyperparameter(0.0, 1.0),
            'dropout_rate_2': Hyperparameter(0.0, 1.0),
            'prelu_alpha_1': Hyperparameter(0.0, 1.0),
            'prelu_alpha_2': Hyperparameter(0.0, 1.0),
            'prelu_alpha_3': Hyperparameter(0.0, 1.0)
            },
        optimizer_params = {
            'lr': Hyperparameter(1e-6, 1e-2), # Learning rate.
            'momentum': Hyperparameter(1e-1, 1e-0), # Parameter that accelerates SGD in the relevant direction and dampens oscillations.
            'weight_decay': Hyperparameter(0.0, 1e-5), # Learning rate decay over each update.
            'nesterov': Hyperparameter(False, True, is_categorical = True) # Whether to apply Nesterov momentum.
            })
    return model_class, optimizer_class, loss_function, train_data, eval_data, test_data, hyper_parameters
    
def setup_fraud():
    model_class = FraudNet
    optimizer_class = torch.optim.SGD
    loss_function = torch.nn.BCELoss()
    # prepare training and testing data
    df = pandas.read_csv('./data/CreditCardFraud/creditcard.csv')
    X = df.iloc[:, :-1].values # extracting features
    y = df.iloc[:, -1].values # extracting labels
    sc = sklearn.preprocessing.StandardScaler()
    X = sc.fit_transform(X)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).float()
    train_data = torch.utils.data.TensorDataset(X_train, Y_train)
    test_data = torch.utils.data.TensorDataset(X_test, Y_test)
    # split training set into training set and validation set
    train_data, eval_data = split_dataset(train_data, 0.9)
    # define hyper-parameter search space
    hyper_parameters = Hyperparameters(
        general_params = None,
        model_params = {
            'dropout_rate_1': Hyperparameter(0.0, 1.0),
            'prelu_alpha_1': Hyperparameter(0.0, 1.0),
            'prelu_alpha_2': Hyperparameter(0.0, 1.0),
            'prelu_alpha_3': Hyperparameter(0.0, 1.0),
            'prelu_alpha_4': Hyperparameter(0.0, 1.0),
            'prelu_alpha_5': Hyperparameter(0.0, 1.0)
            },
        optimizer_params = {
            'lr': Hyperparameter(1e-6, 1e-1), # Learning rate.
            'momentum': Hyperparameter(1e-1, 1e-0), # Parameter that accelerates SGD in the relevant direction and dampens oscillations.
            'weight_decay': Hyperparameter(0.0, 1e-5), # Learning rate decay over each update.
            'nesterov': Hyperparameter(False, True, is_categorical = True) # Whether to apply Nesterov momentum.
            })
    return model_class, optimizer_class, loss_function, train_data, eval_data, test_data, hyper_parameters

def import_user_arguments():
    # import user arguments
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--population_size", type=int, default=5, help="The number of members in the population. Default: 5.")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of batches in which the training set will be divided into.")
    parser.add_argument("--database_path", type=str, default='checkpoints/mnist', help="Directory path to where the checkpoint database is to be located. Default: 'checkpoints/'.")
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
    model_class, optimizer_class, loss_function, train_data, eval_data, test_data, hyper_parameters = setup_mnist()
    # create trainer, evaluator and tester
    trainer = Trainer(
        model_class = model_class,
        optimizer_class = optimizer_class,
        loss_function = loss_function,
        batch_size = args.batch_size,
        train_data = train_data,
        device = args.device,
        verbose = False)
    evaluator = Evaluator(
        model_class = model_class,
        batch_size = args.batch_size,
        test_data = eval_data,
        device = args.device,
        verbose = False)
    tester = Evaluator(
        model_class = model_class,
        batch_size = args.batch_size,
        test_data = test_data,
        device = args.device,
        verbose = False)
    # define controller
    print(f"Creating evolver...")
    steps = 100#2*10**3
    end_criteria = {'steps': steps * 10, 'score': 100.0} #400*10**3
    evolver = ExploitAndExplore(N = args.population_size, exploit_factor = 0.2, explore_factors = (0.8, 1.2))
    #evolver = DifferentialEvolution(N = args.population_size, F = 0.2, Cr = 0.8)
    # create controller
    print(f"Creating controller...")
    controller = Controller(
        population_size=args.population_size,
        hyper_parameters=hyper_parameters,
        trainer=trainer,
        evaluator=evaluator,
        tester=tester,
        evolver=evolver,
        database=database,
        tensorboard_writer=tensorboard_writer,
        step_size=steps,
        evolve_frequency=steps,
        end_criteria=end_criteria,
        device=args.device,
        verbose=args.verbose,
        logging=args.logging)
    # run controller
    print(f"Starting controller...")
    controller.start()
    # analyze results stored in database
    analyzer = Analyzer(database, tester)
    print("Database entries:")
    database.print()
    print("Analyzing population...")
    analyzer.create_plot_files(
        save_directory=database.create_folder("results/plots"),
        n_hyper_parameters=len(hyper_parameters),
        min_score=0,
        max_score=100,
        annotate=False,
        sensitivity=4)
    n_members_to_be_tested = 10
    print(f"Testing the top {n_members_to_be_tested} members on the full test set of {len(test_data)} samples...")
    all_checkpoints = analyzer.test(limit=n_members_to_be_tested)
    best_checkpoint = max(all_checkpoints, key=lambda c: c.test_score)
    print("Results...")
    result = f"Member {best_checkpoint.id} performed best on epoch {best_checkpoint.epochs} / step {best_checkpoint.steps} with an accuracy of {best_checkpoint.test_score:.4f}%"
    database.create_file("results", "best_member.txt").write_text(result)
    with database.create_file("results", "top_members.txt").open('a+') as f:
        for checkpoint in all_checkpoints:
            f.write(str(checkpoint) + "\n")
    print(result)