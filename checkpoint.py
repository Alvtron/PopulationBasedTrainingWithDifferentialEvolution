from hyperparameters import Hyperparameters

class Checkpoint(object):
    '''Class for keeping track of a worker.'''
    def __init__(self, id, hyper_parameters : Hyperparameters, loss_metric : str, eval_metric : str, step_size : int = 1):
        self.id = id
        self.epochs = 0
        self.steps = 0
        self.hyper_parameters = hyper_parameters
        self.model_state = None
        self.optimizer_state = None
        self.step_size = step_size
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss = dict()
        self.time = dict()

    @property
    def score(self):
        return self.loss['eval'][self.eval_metric]

    def __str__(self):
        string = f"Member {self.id:03d}, epoch {self.epochs}, step {self.steps}"
        for loss_group, loss_values in self.loss.items():
            for loss_name, loss_value in loss_values.items():
                string += f", {loss_group}_{loss_name} {loss_value:.5f}"
        return string

    def update(self, checkpoint):
        self.hyper_parameters = checkpoint.hyper_parameters
        self.model_state = checkpoint.model_state
        self.optimizer_state = checkpoint.optimizer_state
        self.loss = dict()
        self.time = dict()
