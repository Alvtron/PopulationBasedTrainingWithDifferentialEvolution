from dataclasses import dataclass
import pandas
import sklearn.preprocessing
import numpy as np
import torch
import torch.utils.data
from torch import from_numpy
from torch.optim import Optimizer
from torch.utils.data import Dataset

from .task import Task
from ..models import hypernet, fraudnet
from ..utils.data import split, random_split, stratified_split
from ..hyperparameters import ContiniousHyperparameter, DiscreteHyperparameter, Hyperparameters
from ..loss import F1, Accuracy, BinaryCrossEntropy
from ..dataset import Datasets


bce = BinaryCrossEntropy()
f1 = F1(classes=2)
accuracy = Accuracy()


class CreditCardFraud(Task):

    def __init__(self):
        pass

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def model_class(self) -> hypernet.HyperNet:
        return fraudnet.FraudNet

    @property
    def optimizer_class(self) -> Optimizer:
        return torch.optim.SGD

    @property
    def hyper_parameters(self) -> Hyperparameters:
        return Hyperparameters(
            model= self.model_class.create_hyper_parameters(),
            optimizer={
                'lr': ContiniousHyperparameter(1e-5, 1e-1),
                'momentum': ContiniousHyperparameter(0.8, 1.0),
                'weight_decay': ContiniousHyperparameter(0.0, 1e-3),
            })

    @property
    def loss_functions(self) -> dict:
        return {
            bce.iso: bce,
            f1.iso: f1,
            accuracy.iso: accuracy
        }

    @property
    def loss_metric(self) -> str:
        return bce.iso

    @property
    def eval_metric(self) -> str:
        return f1.iso

    @property
    def datasets(self) -> Datasets:
        df = pandas.read_csv('./data/CreditCardFraud/creditcard.csv')
        inputs = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        sc = sklearn.preprocessing.StandardScaler()
        torch_inputs = sc.fit_transform(inputs)
        torch_inputs = from_numpy(torch_inputs).float()
        torch_labels = from_numpy(labels).float()
        dataset = torch.utils.data.TensorDataset(torch_inputs, torch_labels)
        # split dataset into training-, testing- and validation set
        unique_labels = set(labels)
        train_data, test_data = stratified_split(dataset, unique_labels, fraction=0.9, random_state=1)
        train_data, eval_data = stratified_split(train_data, unique_labels, fraction=0.9, random_state=1)
        return Datasets(train_data, eval_data, test_data)