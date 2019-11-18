import argparse
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import MnistNet
from member import Member
from database import SharedDatabase
from hyperparameters import Hyperparameter, Hyperparameters
from controller import ExploitAndExplore

mp = torch.multiprocessing.get_context('spawn')

if __name__ == "__main__": 
    # request arguments
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--device", type=str, default='cpu', help="Set processor device ('cpu' or 'gpu' or 'cuda'). GPU is not supported on windows for PyTorch multiproccessing. Default: 'cpu'.")
    parser.add_argument("--population_size", type=int, default=5, help="The number of members in the population. Default: 5.")
    parser.add_argument("--batch_size", type=int, default= 512, help="The number of batches in which the training set will be divided into.")
    parser.add_argument("--database_path", type=str, default='checkpoints', help="Directory path to where the checkpoint database is to be located. Default: 'checkpoints/'.")
    # import arguments
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() and not os.name == 'nt' else 'cpu'
    population_size = args.population_size
    batch_size = args.batch_size
    database_path = args.database_path
    # prepare database
    manager = mp.Manager()
    data = manager.dict()
    database = SharedDatabase(directory_path = 'checkpoints', shared_memory_dict = data)
    # prepare training and testing data
    train_data_path = test_data_path = './data'
    train_data = MNIST(train_data_path, True, transforms.ToTensor(), download=True)
    test_data = MNIST(test_data_path, False, transforms.ToTensor(), download=True)
    # define controller
    controller = ExploitAndExplore(
        exploit_factor = 0.2,
        explore_factors = (0.8, 1.2),
        frequency = 10,
        end_criteria = {
            'epoch': 50,
            #'score': 99.5
            })
    # define hyper-parameter search space
    hyper_parameters = Hyperparameters(
        general_params = None,
        model_params = None,
        optimizer_params = {
            'lr': Hyperparameter(1e-6, 1e-0), # Learning rate.
            'momentum': Hyperparameter(1e-1, 1e-0), # Parameter that accelerates SGD in the relevant direction and dampens oscillations.
            'weight_decay': Hyperparameter(0.0, 1e-5), # Learning rate decay over each update.
            #'dampening': Hyperparameter(1e-10, 1e-1), # Dampens oscillation from Nesterov momentum.
            'nesterov': Hyperparameter(False, True, is_categorical = True) # Whether to apply Nesterov momentum.
            })
    # create members
    members = [
        Member(
            id = id,
            model_class = MnistNet,
            optimizer_class = torch.optim.SGD,
            controller = controller,
            batch_size = batch_size,
            hyper_parameters = hyper_parameters,
            loss_function = torch.nn.CrossEntropyLoss(),
            train_data = train_data,
            test_data = test_data,
            database = database,
            device = device,
            verbose = True)
        for id in range(population_size)]
    # In multiprocessing, processes are spawned by creating a Process object and then calling its start() method
    [w.start() for w in members]
    # join() blocks the calling thread until the process whose join() method is called terminates or until the optional timeout occurs.
    [w.join() for w in members]
    # print database and best member
    print("---")
    database.print()
    print("---")
    all_checkpoints = database.to_list()
    best_checkpoint = max(all_checkpoints, key=lambda c: c.score)
    print(f"Member {best_checkpoint.id} performed best on epoch {best_checkpoint.epochs} with an accuracy of {best_checkpoint.score:.2f}%")
