import argparse
import os
import math
import pathlib
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import Net
from Member import Member
from checkpoint import Checkpoint
from hyperparameters import Hyperparameter

mp = torch.multiprocessing.get_context('spawn')
#torch.set_default_tensor_type(torch.cuda.FloatTensor)

def exploit_and_explore(member, population):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--device", type=str, default='cpu', help="Set processor device ('cpu' or 'gpu' or 'cuda'). GPU is not supported on windows for PyTorch multiproccessing.")
    parser.add_argument("--population_size", type=int, default=12, help="The number of members in the population.")
    parser.add_argument("--batch_size", type=int, default=32, help="How many samples to process at once on the CPU/GPU. Check your specification.")
    parser.add_argument("--max_epoch", type=int, default=40, help="")
    parser.add_argument("--clear_checkpoints", type=bool, default=True, help="")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() and not os.name == 'nt' else 'cpu'
    population_size = args.population_size
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    clear_checkpoints = args.clear_checkpoints
    # create checkpoint folder
    if clear_checkpoints:
        for f in [ f for f in os.listdir('checkpoints') if f.endswith(".pth") ]:
            os.remove(os.path.join('checkpoints', f))
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
                    'lr': Hyperparameter(1e-5, 1e-1), # Learning rate.
                    'momentum': Hyperparameter(1e-10, 1.0), # Parameter that accelerates SGD in the relevant direction and dampens oscillations.
                    'weight_decay': Hyperparameter(0.0, 1e-1), # Learning rate decay over each update.
                    'dampening': Hyperparameter(1e-10, 1e-1), # Dampens oscillation from Nesterov momentum.
                    'nesterov': Hyperparameter(False, True) # Whether to apply Nesterov momentum.
                    }
                },
            mutation_function = exploit_and_explore,
            batch_size = batch_size,
            max_epoch = max_epoch,
            train_data = train_data,
            test_data = test_data,
            population_size = population_size,
            device = device,
            verbose = 1)
        for id in range(population_size)]
    # In multiprocessing, processes are spawned by creating a Process object and then calling its start() method
    [w.start() for w in members]
    # join() blocks the calling thread until the process whose join() method is called terminates or until the optional timeout occurs.
    [w.join() for w in members]
    checkpoints = []
    for id in range(population_size):
        checkpoint_path = f"checkpoints/w{id:03d}.pth"
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            checkpoints.append(checkpoint)
    checkpoints.sort(key=lambda w: w.score, reverse=True)
    best_member = checkpoints[0]
    print(f"Worker {best_member.id} performed best with an accuracy of {best_member.score:.2f}%")
