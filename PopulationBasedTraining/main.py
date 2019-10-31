import argparse
import operator
import os
import time
import math
import pathlib
import random
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as _mp
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from model import Net
from dataclasses import dataclass

mp = _mp.get_context('spawn')
#torch.set_default_tensor_type(torch.cuda.FloatTensor)

@dataclass
class Checkpoint:
    '''Class for keeping track of a worker.'''
    id: int
    epoch: int
    model_state: dict
    optimizer_state: dict
    hyperparameters: dict
    batch_size: int
    score: float

    def update(self, checkpoint):
        self.epoch = checkpoint.epoch
        self.model_state = checkpoint.model_state
        self.optimizer_state = checkpoint.optimizer_state
        self.hyperparameters = checkpoint.hyperparameters
        self.batch_size = checkpoint.batch_size
        score = None

    def __str__(self):
        return f"Worker {self.id} ({self.score}%)"

def exploit(worker, population):
    elitist = random.choice(population)
    print(f"Exploiting w{elitist.id} to w{worker.id}")
    worker.update(elitist)

def explore(worker, verbose = True, perturb_factors = (1.2, 0.8)):
    if verbose:
        print(f"Exploring w{worker.id}")
    # exploring optimizer
    for hyperparameter_name in worker.hyperparameters['optimizer']:
        for parameter_group in worker.optimizer_state['param_groups']:
            perturb = np.random.choice(perturb_factors)
            parameter_group[hyperparameter_name] *= perturb
    # exploring batch_size
    if worker.hyperparameters['batch_size']:
        perturb = np.random.choice(perturb_factors)
        worker.batch_size = int(np.ceil(perturb * worker.batch_size))

class Worker(mp.Process):
    def __init__(self, id, hyperparameters, batch_size, max_epoch, train_data, test_data, population_size, device, verbose):
        super().__init__()
        self.id = id
        self.hyperparameters = hyperparameters
        self.epoch = 0
        self.population_size = population_size
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.device = device
        self.model = Net().to(device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=np.random.choice(np.logspace(-5, 0, base=10)),
            momentum=np.random.choice(np.linspace(0.1, .9999)))
        self.score = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_data = train_data
        self.test_data = test_data
        self.verbose = verbose

    def run(self):
        #self.load_checkpoint_from_path(f"checkpoints/w{self.id:03d}.pth")
        while self.epoch < self.max_epoch: #not end of training
            self.train() # step
            self.score = self.eval() # eval
            if True: # if ready-condition
                checkpoint = self.create_checkpoint()
                population = self.load_population()
                # sort members
                population.sort(key=lambda x: x.score, reverse=True)
                # set number of elitists
                n_elitists = math.floor(len(population) * 0.4)
                if n_elitists > 0 and all(member.id != self.id for member in population[:n_elitists]):
                    # exploit
                    exploit(checkpoint, population[:n_elitists])
                # explore
                explore(checkpoint)
                self.load_checkpoint(checkpoint)
                self.score = self.eval()
            if self.verbose > 0:
                print(f"Score of w{self.id}: {self.score:.2f}% (e{self.epoch})")
            self.epoch += 1
            # save to population
            self.save_checkpoint(f"checkpoints/w{self.id:03d}.pth")
        print(f"Worker {self.id} is finished.")

    def create_checkpoint(self):
        """Create checkpoint from worker state."""
        checkpoint = Checkpoint(
            self.id,
            self.epoch,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.hyperparameters,
            self.batch_size,
            self.score)
        return checkpoint

    def save_checkpoint(self, checkpoint_path):
        """Save worker state to checkpoint_path."""
        checkpoint = self.create_checkpoint()
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint_from_file(self, checkpoint_path):
        """Load worker state from checkpoint path."""
        assert os.path.isfile(checkpoint_path), f"checkpoint file does not exist on path: {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path)
        self.load_checkpoint(checkpoint)
        
    def load_checkpoint(self, checkpoint):
        """Load worker state from checkpoint object."""
        self.id = checkpoint.id
        self.epoch = checkpoint.epoch
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.batch_size = checkpoint.batch_size
        self.score = checkpoint.score
        # set dropout and batch normalization layers to evaluation mode before running inference
        self.model.eval()

    def load_population(self):
        """Load population as list of checkpoints."""
        checkpoints = []
        for id in range(self.population_size):
            checkpoint_path = f"checkpoints/w{id:03d}.pth"
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                checkpoints.append(checkpoint)
        return checkpoints

    def train(self):
        """Train the model on the provided training set."""
        self.model.train()
        dataloader = None
        if self.verbose == 1:
            print(f"Training w{self.id}... (e{self.epoch})")
        if self.verbose < 2:
            dataloader = DataLoader(self.train_data, self.batch_size, True)
        else:
            dataloader = tqdm.tqdm(
            DataLoader(self.train_data, self.batch_size, True),
            desc = f"Training w{self.id} (e{self.epoch})",
            ncols = 80,
            leave = True)
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.loss_fn(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self):
        """Evaluate model on the provided validation or test set."""
        self.model.eval()
        dataloader = None
        if self.verbose == 1:
            print(f"Evaluating w{self.id}... (e{self.epoch})")
        if self.verbose < 2:
            dataloader = DataLoader(self.test_data, self.batch_size, True)
        else:
            dataloader = tqdm.tqdm(
                DataLoader(self.test_data, self.batch_size, True),
                desc = f"Evaluating w{self.id} (epoch {self.epoch})",
                ncols = 80,
                leave = True)

        correct = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            pred = output.argmax(1)
            correct += pred.eq(y).sum().item()
        accuracy = 100. * correct / (len(dataloader) * self.batch_size)
        return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--device", type=str, default='cpu', help="Set processor device ('cpu' or 'gpu' or 'cuda'). GPU is not supported on windows for PyTorch multiproccessing.")
    parser.add_argument("--population_size", type=int, default=10, help="The number of members in the population.")
    parser.add_argument("--batch_size", type=int, default=32, help="How many samples to process at once on the CPU/GPU. Check your specification.")
    parser.add_argument("--max_epoch", type=int, default=10, help="")
    parser.add_argument("--clear_checkpoints", type=bool, default=True, help="")
    args = parser.parse_args()
    #mp.set_start_method("spawn")
    #mp = mp.get_context('forkserver')
    device = args.device if torch.cuda.is_available() and not os.name == 'nt' else 'cpu'
    population_size = args.population_size
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    clear_checkpoints = args.clear_checkpoints
    # create checkpoint folder
    if clear_checkpoints:
        for f in [ f for f in os.listdir('checkpoints') if f.endswith(".pth") ]:
            os.remove(os.path.join('checkpoints', f))
    # Returns a process shared queue implemented using a pipe and a few locks/semaphores. When a process first puts an item on the queue a feeder thread is started which transfers objects from a buffer into the pipe.
    population = mp.Queue(maxsize=population_size)
    hyperparameters = {'optimizer': ["lr", "momentum"], "batch_size": True}
    train_data_path = test_data_path = './data'
    train_data = MNIST(train_data_path, True, transforms.ToTensor(), download=True)
    test_data = MNIST(test_data_path, False, transforms.ToTensor(), download=True)
    # create workers
    workers = [
        Worker(
            id,
            hyperparameters,
            batch_size,
            max_epoch,
            train_data,
            test_data,
            population_size,
            device,
            1)
        for id in range(population_size)]

    # In multiprocessing, processes are spawned by creating a Process object and then calling its start() method
    [w.start() for w in workers]
    # join() blocks the calling thread until the process whose join() method is called terminates or until the optional timeout occurs.
    [w.join() for w in workers]
    checkpoints = []
    for id in range(population_size):
        checkpoint_path = f"checkpoints/w{id:03d}.pth"
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            checkpoints.append(checkpoint)
    checkpoints.sort(key=lambda w: w.score, reverse=True)
    best_worker = checkpoints[0]
    print(f"Worker {best_worker.id} performed best with an accuracy of {best_worker.score:.2f}%")
