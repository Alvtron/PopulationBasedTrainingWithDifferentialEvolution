import unittest
import copy
import itertools

import torch

import pbt.utils.data
from pbt.utils.iterable import unwrap_iterable
from pbt.models.lenet5 import LeNet5
from pbt.database import ReadOnlyDatabase
from pbt.analyze import Analyzer 
from pbt.loss import CategoricalCrossEntropy, Accuracy, F1
from pbt.evolution import DifferentialEvolution, SHADE, LSHADE
from pbt.hyperparameters import Hyperparameters, ContiniousHyperparameter, DiscreteHyperparameter
from pbt.member import Checkpoint, Generation
from pbt.task.mnist import MnistKnowledgeSharing
from pbt.worker_pool import WorkerPool

class TestEvolving(unittest.TestCase):
    def logger(self, message, checkpoint = None):
        return

    def setUp(self):
        self.population_size = 10
        loss_metric = 'cce'
        eval_metric = 'acc'
        loss_functions = {
            'cce': CategoricalCrossEntropy(),
            'acc': Accuracy()
        }
        hp = Hyperparameters(
            optimizer={
                'lr': ContiniousHyperparameter(0.000001, 0.1, value=0.01),
                'momentum': ContiniousHyperparameter(0.001, 0.9, value=0.5)
            })
        self.members = Generation()
        for i in range(self.population_size):
            member = Checkpoint(
                uid=i,
                parameters=hp,
                loss_metric=loss_metric,
                eval_metric=eval_metric,
                minimize=loss_functions[eval_metric].minimize)
            member.loss = {
                'eval': {eval_metric:i*10},
                'train': {loss_metric:i/10}}
            self.members.append(member)

    def test_de(self):
        evolver = DifferentialEvolution()
        old_members = self.members
        for member in self.members:
            evolver.on_spawn(member, self.logger)
        evolver.on_generation_start(member, self.logger)
        canidates = evolver.on_evolve(old_members, self.logger)
        new_members = Generation()
        for canidate in canidates:
            best = evolver.on_evaluate(canidate, self.logger)
            new_members.append(best)
        evolver.on_generation_end(old_members, self.logger)
        self.assertTrue(all(id(old) != id(new) for old, new in zip(old_members, new_members)))
        self.assertTrue(all(id(old.parameters) != id(new.parameters)  for old, new in zip(old_members, new_members)))
    
    def test_shade(self):
        evolver = SHADE(self.population_size)
        old_members = self.members
        for member in self.members:
            evolver.on_spawn(member, self.logger)
        evolver.on_generation_start(member, self.logger)
        canidates = evolver.on_evolve(old_members, self.logger)
        new_members = Generation()
        for canidate in canidates:
            best = evolver.on_evaluate(canidate, self.logger)
            new_members.append(best)
        evolver.on_generation_end(old_members, self.logger)
        self.assertTrue(all(id(old) != id(new) for old, new in zip(old_members, new_members)))
        self.assertTrue(all(id(old.parameters) != id(new.parameters)  for old, new in zip(old_members, new_members)))

    def test_lshade(self):
        evolver = LSHADE(self.population_size, 100)
        old_members = copy.deepcopy(self.members)
        for member in old_members:
            evolver.on_spawn(member, self.logger)
        old_members_copy = copy.deepcopy(old_members)
        evolver.on_generation_start(member, self.logger)
        canidates = evolver.on_evolve(old_members, self.logger)
        new_members = Generation()
        for canidate in canidates:
            best = evolver.on_evaluate(canidate, self.logger)
            new_members.append(best)
        evolver.on_generation_end(old_members, self.logger)
        self.assertTrue(all(id(old) != id(new) for old, new in zip(old_members, new_members)))
        self.assertTrue(all(id(old.parameters) != id(new.parameters) for old, new in zip(old_members, new_members)))
        self.assertTrue(all(old.parameters != original_copy.parameters for old, original_copy in zip(old_members, self.members)))
        self.assertTrue(all(old.parameters == old_copy.parameters for old, old_copy in zip(old_members, old_members_copy)))