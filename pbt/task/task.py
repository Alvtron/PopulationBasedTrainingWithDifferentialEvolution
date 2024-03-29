from abc import ABC
from typing import Dict

from torch.optim import Optimizer
from torch.utils.data import Dataset

from ..models.hypernet import HyperNet
from ..hyperparameters import Hyperparameters
from ..loss import _Loss
from ..dataset import Datasets

class Task(ABC):
    """
    Base class for all tasks.
    """

    @property
    def num_classes(self) -> int:
        raise NotImplementedError()

    @property
    def model_class(self) -> HyperNet:
        raise NotImplementedError()

    @property
    def optimizer_class(self) -> Optimizer:
        raise NotImplementedError()

    @property
    def hyper_parameters(self) -> Hyperparameters:
        """Define hyper-parameter search space. """
        raise NotImplementedError()

    @property
    def loss_functions(self) -> Dict[str, _Loss]:
        raise NotImplementedError()

    @property
    def loss_metric(self) -> str:
        raise NotImplementedError()

    @property
    def eval_metric(self) -> str:
        raise NotImplementedError()

    @property
    def datasets(self) -> Datasets:
        raise NotImplementedError()