import argparse
import os
import math
import torch
import torchvision
import torch.utils.data
import pandas
import sklearn.preprocessing
import sklearn.model_selection
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import MnistNet, FraudNet
from member import Member
from database import SharedDatabase
from hyperparameters import Hyperparameter, Hyperparameters
from controller import ExploitAndExplore, DifferentialEvolution, ParticleSwarm
from trainer import Trainer
from evaluator import Evaluator
from analyze import Analyzer

mp = torch.multiprocessing.get_context('spawn')

def split_dataset(dataset, fraction):
        assert 0.0 <= fraction <= 1.0, f"The provided fraction must be between 0.0 and 1.0!"
        dataset_length = len(dataset)
        first_set_length = math.floor(fraction * dataset_length)
        second_set_length = dataset_length - first_set_length
        first_set, second_set = torch.utils.data.random_split(dataset, (first_set_length, second_set_length))
        return first_set, second_set

def setup_mnist(population_size, batch_size, step_size, controller, database, device, verbose, logging):
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
    train_data, eval_data = split_dataset(train_data, 0.9)
    # create trainer and evaluator
    trainer = Trainer(
        model_class = MnistNet,
        optimizer_class = torch.optim.SGD,
        loss_function = torch.nn.CrossEntropyLoss(),
        batch_size = batch_size,
        train_data = train_data,
        device = device,
        verbose = False)
    evaluator = Evaluator(
        model_class = MnistNet,
        batch_size = batch_size,
        test_data = eval_data,
        device = device,
        verbose = False)
    tester = Evaluator(
        model_class = MnistNet,
        batch_size = batch_size,
        test_data = test_data,
        device = device,
        verbose = False)
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
            #'weight_decay': Hyperparameter(0.0, 1e-5), # Learning rate decay over each update.
            'nesterov': Hyperparameter(False, True, is_categorical = True) # Whether to apply Nesterov momentum.
            })
    # create members
    members = [
        Member(
            id = id,
            controller = controller,
            hyper_parameters = hyper_parameters,
            trainer = trainer,
            evaluator = evaluator,
            database = database,
            step_size = step_size,
            device = device,
            verbose = verbose,
            logging = logging)
        for id in range(population_size)]
    analyzer = Analyzer(database, tester)
    return members, analyzer
    
def setup_fraud(population_size, batch_size, step_size, controller, database, device, verbose, logging):
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
    # create trainer and evaluator
    trainer = Trainer(
        model_class = FraudNet,
        optimizer_class = torch.optim.SGD,
        loss_function = torch.nn.BCELoss(),
        batch_size = batch_size,
        train_data = train_data,
        device = device,
        verbose = False)
    evaluator = Evaluator(
        model_class = FraudNet,
        batch_size = batch_size,
        test_data = eval_data,
        device = device,
        verbose = False)
    tester = Evaluator(
        model_class = FraudNet,
        batch_size = batch_size,
        test_data = test_data,
        device = device,
        verbose = False)
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
    # create members
    members = [
        Member(
            id = id,
            controller = controller,
            hyper_parameters = hyper_parameters,
            trainer = trainer,
            evaluator = evaluator,
            database = database,
            step_size = step_size,
            device = device,
            verbose = verbose,
            logging = logging)
        for id in range(population_size)]
    analyzer = Analyzer(database, tester)
    return members, analyzer

if __name__ == "__main__": 
    # request arguments
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--device", type=str, default='cpu', help="Set processor device ('cpu' or 'gpu' or 'cuda'). GPU is not supported on windows for PyTorch multiproccessing. Default: 'cpu'.")
    parser.add_argument("--population_size", type=int, default=5, help="The number of members in the population. Default: 5.")
    parser.add_argument("--batch_size", type=int, default= 64, help="The number of batches in which the training set will be divided into.")
    parser.add_argument("--database_path", type=str, default='checkpoints', help="Directory path to where the checkpoint database is to be located. Default: 'checkpoints/'.")
    # import arguments
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() and not os.name == 'nt' else 'cpu'
    population_size = args.population_size
    batch_size = args.batch_size
    database_path = args.database_path
    # prepare database
    database_directory_path = 'checkpoints/mnist'
    manager = mp.Manager()
    shared_memory_dict = manager.dict()
    database = SharedDatabase(directory_path = database_directory_path, shared_memory_dict = shared_memory_dict)
    # define controller
    steps = 100 #2*10**3
    end_steps_criterium = 10*steps #400*10**3
    #controller = ExploitAndExplore(exploit_factor = 0.2, explore_factors = (0.8, 1.2), frequency = steps, end_criteria = {'steps': end_steps_criterium, 'score': 100.0})
    controller = DifferentialEvolution(N = population_size, F = 0.2, Cr = 0.8, frequency = steps, end_criteria = {'steps': end_steps_criterium, 'score': 100.0})
    # create members
    members, analyzer = setup_mnist(
        population_size=population_size,
        batch_size=batch_size,
        step_size=steps,
        controller=controller,
        database=database,
        device=device,
        verbose=True,
        logging=True)
    # spawn members
    [m.start() for m in members]
    # block the calling thread until the members are finished
    [m.join() for m in members]
    # print database and best member
    print("Database entries:")
    database.print()
    print("Analyzing population...")
    all_checkpoints = analyzer.test(limit=25)
    best_checkpoint = max(all_checkpoints, key=lambda c: c.score)
    analyzer.plot_hyperparams(0)
    print("Results...")
    result = f"Member {best_checkpoint.id} performed best on epoch {best_checkpoint.epochs} / step {best_checkpoint.steps} with an accuracy of {best_checkpoint.score:.4f}%"
    database.save_to_file("results.txt", result)
    print(result)
