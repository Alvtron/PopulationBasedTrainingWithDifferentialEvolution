import argparse
import os
import math
import pathlib
import random
import operator
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import Net
from Member import Member
from database import Checkpoint, Database
from hyperparameters import Hyperparameter

mp = torch.multiprocessing.get_context('spawn')
#torch.set_default_tensor_type(torch.cuda.FloatTensor)

def end_training_condition(checkpoint, database):
    # end if 10 epochs have passed
    if checkpoint.epoch > 10:
        return True
    all_checkpoints = database.to_list()
    if not all_checkpoints:
        return False
    best_score = max(c.score for c in all_checkpoints)
    # end if score is above 99%
    if best_score > 99.00:
        return True
    return False

def ready_condition(current, database):
    """True every 5th epoch"""
    history = database.get_entries(current.id)
    return current.epoch % 5 == 0

def exploit_and_explore(member, database):
    population = database.get_latest()
    # sort members
    population.sort(key=lambda m: m.score, reverse=True)
    # set number of elitists
    n_elitists = math.floor(len(population) * 0.4)
    if n_elitists > 0 and all(m.id != member.id for m in population[:n_elitists]):
        # exploit
        exploit(member, population[:n_elitists])
    # explore
    explore(member)

def exploit(member, population):
    random.seed(member.id)
    elitist = random.choice(population)
    print(f"Exploit: w{member.id} <-- w{elitist.id}")
    member.update(elitist)

def explore(member, perturb_factors = (1.2, 0.8)):
    print(f"Explore: w{member.id}")
    # exploring optimizer
    for hyperparameter_name, hyperparameter in member.hyperparameters['optimizer'].items():
        for parameter_group in member.optimizer_state['param_groups']:
            hyperparameter.perturb(perturb_factors)
            parameter_group[hyperparameter_name] = hyperparameter.value
    # exploring batch_size
    if member.hyperparameters['batch_size']:
        member.hyperparameters['batch_size'].perturb(perturb_factors)
        member.batch_size = member.hyperparameters['batch_size'].value

def prepare_checkpoints_folder(clear_checkpoints : bool = True):
    directory_path = 'checkpoints'
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    if clear_checkpoints:
        for f in [ f for f in os.listdir(directory_path) if f.endswith(".pth") ]:
            os.remove(os.path.join(directory_path, f))

if __name__ == "__main__":
    # request arguments
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--device", type=str, default='cpu', help="Set processor device ('cpu' or 'gpu' or 'cuda'). GPU is not supported on windows for PyTorch multiproccessing.")
    parser.add_argument("--population_size", type=int, default=10, help="The number of members in the population.")
    parser.add_argument("--clear_checkpoints", type=bool, default=True, help="")
    # import arguments
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() and not os.name == 'nt' else 'cpu'
    population_size = args.population_size
    clear_checkpoints = args.clear_checkpoints
    # prepare checkpoint folder
    prepare_checkpoints_folder(clear_checkpoints)
    database = Database(directory_path = 'checkpoints')
    # prepare training and testing data
    train_data_path = test_data_path = './data'
    train_data = MNIST(train_data_path, True, transforms.ToTensor(), download=True)
    test_data = MNIST(test_data_path, False, transforms.ToTensor(), download=True)
    # create members
    members = [
        Member(
            id = id,
            model = Net().to(device),
            optimizer = torch.optim.SGD,
            hyperparameters = {
                'batch_size': Hyperparameter(1, 128),
                'optimizer': {
                    'lr': Hyperparameter(1e-6, 1e-1), # Learning rate.
                    'momentum': Hyperparameter(1e-10, 1.0), # Parameter that accelerates SGD in the relevant direction and dampens oscillations.
                    'weight_decay': Hyperparameter(0.0, 1e-1), # Learning rate decay over each update.
                    'dampening': Hyperparameter(1e-10, 1e-1), # Dampens oscillation from Nesterov momentum.
                    'nesterov': Hyperparameter(False, True) # Whether to apply Nesterov momentum.
                    }
                },
            mutation_function = exploit_and_explore,
            ready_condition = ready_condition,
            end_training_condition = end_training_condition,
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
    print(f"Member {best_checkpoint.id} performed best on epoch {best_checkpoint.epoch} with an accuracy of {best_checkpoint.score:.2f}%")
