from dataclasses import dataclass

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