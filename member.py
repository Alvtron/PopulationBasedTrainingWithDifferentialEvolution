import copy

from hyperparameters import Hyperparameters
from abc import abstractmethod 

class MemberState(object):
    '''Base class for member states.'''
    def __init__(self, id, hyper_parameters : Hyperparameters, minimize : bool):
        self.id = id
        self.hyper_parameters = hyper_parameters
        self.minimize = minimize

    @abstractmethod
    def score(self):
        pass

    def __str__(self):
        return f"Member {self.id:03d}"

    def __lt__(self, other):
        if isinstance(other, (float, int)):
            return self.score() > other if self.minimize else self.score() < other
        else:
            return self.score() > other.score() if self.minimize else self.score() < other.score()

    def __gt__(self, other):
        if isinstance(other, (float, int)):
            return self.score() < other if self.minimize else self.score() > other
        else:
            return self.score() < other.score()if self.minimize else self.score() > other.score()

    def __le__(self, other):
        if isinstance(other, (float, int)):
            return self.score() <= other if self.minimize else self.score() >= other
        else:
            return self.score() <= other.score()if self.minimize else self.score() >= other.score()

    def __ge__(self, other):
        if isinstance(other, (float, int)):
            return self.score() >= other if self.minimize else self.score() <= other
        else:
            return self.score() >= other.eval_value if self.minimize else self.score() <= other.score()

    def update(self, other):
        self.hyper_parameters = other.hyper_parameters

    def copy(self):
        member_copy = copy.deepcopy(self)
        member_copy.id = None
        return member_copy

class Checkpoint(MemberState):
    '''Class for keeping track of a checkpoint.'''
    def __init__(self, id, hyper_parameters : Hyperparameters, loss_metric : str, eval_metric : str, minimize : bool, step_size : int = 1):
        super().__init__(id, hyper_parameters, minimize)
        self.epochs = 0
        self.steps = 0
        self.model_state = None
        self.optimizer_state = None
        self.step_size = step_size
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss = dict()
        self.time = dict()

    def score(self):
        return self.loss['eval'][self.eval_metric]

    def __str__(self):
        return super().__str__() + f", epoch {self.epochs:03d}, step {self.steps:05d}"

    def peformance_details(self):
        strings = list()
        for loss_group, loss_values in self.loss.items():
            for loss_name, loss_value in loss_values.items():
                strings.append(f"{loss_group}_{loss_name} {loss_value:.5f}")
        return ", ".join(strings)

    def update(self, checkpoint):
        self.hyper_parameters = checkpoint.hyper_parameters
        self.model_state = checkpoint.model_state
        self.optimizer_state = checkpoint.optimizer_state
        self.loss = dict()
        self.time = dict()
