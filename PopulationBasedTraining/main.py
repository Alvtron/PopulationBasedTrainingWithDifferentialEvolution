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

@dataclass
class Checkpoint:
    '''Class for keeping track of a worker.'''
    id: int
    model_state: dict
    optimizer_state: dict
    hyperparameters: dict
    batch_size: int
    score: float

def exploit(worker, population, n_top_members):
    print(f"exploiting for worker {worker.id}")
    index = random.randrange(0, n_top_members, 1)
    top_member = sorted_members[index]
    worker = top_member
    explore(worker, population)

def explore(worker, perturb_factors = (1.2, 0.8)):
    print(f"exploring for worker {worker.id}")
    for hyperparam_name in worker.hyperparameters['optimizer']:
        for param_group in worker.optimizer_state['param_groups']:
            perturb = np.random.choice(perturb_factors)
            param_group[hyperparam_name] *= perturb
    if worker.hyperparameters['batch_size']:
        perturb = np.random.choice(perturb_factors)
        worker.batch_size = int(np.ceil(perturb * worker.batch_size))

class Worker(mp.Process):
    def __init__(self, id, hyperparameters, batch_size, epoch, max_epoch, train_data, test_data, population_size, device):
        super().__init__()
        self.id = id
        self.hyperparameters = hyperparameters
        self.epoch = epoch
        self.population_size = population_size
        self.finish_tasks = finish_tasks
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
        self.task_id = None

    def run(self):
        while True: #not end of training
            if self.epoch.value > self.max_epoch:
                break
            checkpoint_path = f"checkpoints/task-{self.id:03d}.pth"
            #self.load_checkpoint_from_path(checkpoint_path)
            
            try:
                self.train() # step
                self.score = self.eval() # eval
                if True: # if ready-condition
                    checkpoint = self.create_checkpoint()
                    population = self.load_population()
                    n_checkpoints = len(population)
                    # sort members
                    sorted_members = sorted(population, key=operator.attrgetter('score'))
                    # set number of elitists
                    n_top_members = math.floor(n_checkpoints * 0.2)
                    if n_checkpoints > 0 and self.score < sorted_members[n_top_members].score:
                        # exploit
                        exploit(checkpoint, sorted_members, n_top_members)
                    else:
                        # explore
                        explore(checkpoint)
                        self.load_checkpoint(checkpoint)
                        self.score = self.eval()
                    
                # save to population
                self.save_checkpoint(checkpoint_path)
            except KeyboardInterrupt:
                break

    def create_checkpoint(self):
        """Create checkpoint from worker state."""
        checkpoint = Checkpoint(
            self.id,
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

    def load_checkpoint_from_path(self, checkpoint_path):
        """Load worker state from checkpoint path."""
        assert os.path.isfile(checkpoint_path), f"checkpoint file does not exist on path: {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path)
        self.load_checkpoint(checkpoint)
        
    def load_checkpoint(self, checkpoint):
        """Load worker state from checkpoint object."""
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.batch_size = checkpoint.batch_size
        self.score = checkpoint.score

    def load_population(self):
        """Load population as list of checkpoints."""
        checkpoints = []
        for id in range(self.population_size):
            checkpoint_path = f"checkpoints/task-{id:03d}.pth"
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                checkpoints.append(checkpoint)
        return checkpoints

    def train(self):
        """Train the model on the provided training set."""
        self.model.train()
        dataloader = tqdm.tqdm(
            DataLoader(self.train_data, self.batch_size, True),
            desc='Train (task {})'.format(self.id),
            ncols=80, leave=True)
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
        dataloader = tqdm.tqdm(
            DataLoader(self.train_data, self.batch_size, True),
            desc=f"Eval (task {self.id})",
            ncols=80,
            leave=True)
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
    parser.add_argument("--population_size", type=int, default=4, help="The number of members in the population.")
    parser.add_argument("--batch_size", type=int, default=32, help="How many samples to process at once on the CPU/GPU. Check your specification.")
    parser.add_argument("--max_epoch", type=int, default=10, help="")
    args = parser.parse_args()

    #mp.set_start_method("spawn")
    #mp = mp.get_context('forkserver')

    device = args.device if torch.cuda.is_available() and not os.name == 'nt' else 'cpu'
    population_size = args.population_size
    batch_size = args.batch_size
    max_epoch = args.max_epoch

    manager = mp.Manager()
    shared_memory = manager.dict()

    pathlib.Path('checkpoints').mkdir(exist_ok=True)
    # Returns a process shared queue implemented using a pipe and a few locks/semaphores. When a process first puts an item on the queue a feeder thread is started which transfers objects from a buffer into the pipe.
    population = mp.Queue(maxsize=population_size)
    finish_tasks = mp.Queue(maxsize=population_size)
    # Data stored in a shared memory map
    epoch = mp.Value('i', 0)
    for i in range(population_size):
        # Put a dict of scores into the queue
        population.put(dict(id=i, score=0))
    hyper_params = {'optimizer': ["lr", "momentum"], "batch_size": True}
    train_data_path = test_data_path = './data'

    train_data = MNIST(train_data_path, True, transforms.ToTensor(), download=True)
    test_data = MNIST(test_data_path, False, transforms.ToTensor(), download=True)
    
    workers = []

    for id in range(population_size):
        workers.append(Worker(
            id,
            hyper_params,
            batch_size,
            epoch,
            max_epoch,
            train_data,
            test_data,
            population_size,
            device))

    # In multiprocessing, processes are spawned by creating a Process object and then calling its start() method
    [w.start() for w in workers]
    # join() blocks the calling thread until the process whose join() method is called terminates or until the optional timeout occurs.
    [w.join() for w in workers]

    task = []
    while not finish_tasks.empty():
        task.append(finish_tasks.get())
    while not population.empty():
        task.append(population.get())
    task = sorted(task, key=lambda x: x['score'], reverse=True)
    print('best score on', task[0]['id'], 'is', task[0]['score'])
