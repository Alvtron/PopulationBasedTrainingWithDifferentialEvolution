import os
import tqdm
import torch
from torch.utils.data import DataLoader
from checkpoint import Checkpoint

mp = torch.multiprocessing.get_context('spawn')

class Member(mp.Process):
    ''' A individual member in the population '''
    def __init__(self, id, model, optimizer, hyperparameters, mutation_function, batch_size, max_epoch, train_data, test_data, population_size, device, verbose):
        super().__init__()
        self.id = id
        self.hyperparameters = hyperparameters
        self.mutation_function = mutation_function
        self.population_size = population_size
        self.epoch = 0
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.device = device
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr = 0.1, momentum = 0.9)
        self.score = None
        self.loss_fn = torch.nn.CrossEntropyLoss()
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
                self.mutation_function(checkpoint, population)
                self.load_checkpoint(checkpoint)
                self.score = self.eval()
            if self.verbose > 0:
                print(f"Score of w{self.id}: {self.score:.2f}% (e{self.epoch})")
            self.epoch += 1
            # save to population
            self.save_checkpoint(f"checkpoints/w{self.id:03d}.pth")
        print(f"Worker {self.id} is finished.")

    def create_checkpoint(self):
        """Create checkpoint from member state."""
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
        """Save member state to checkpoint_path."""
        checkpoint = self.create_checkpoint()
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint_from_file(self, checkpoint_path):
        """Load member state from checkpoint path."""
        assert os.path.isfile(checkpoint_path), f"checkpoint file does not exist on path: {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path)
        self.load_checkpoint(checkpoint)
        
    def load_checkpoint(self, checkpoint):
        """Load member state from checkpoint object."""
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