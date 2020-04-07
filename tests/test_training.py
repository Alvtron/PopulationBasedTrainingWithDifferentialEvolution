import unittest
import shutil
import itertools

import torch

import pbt.utils.data
from pbt.utils.iterable import unwrap_iterable
from pbt.models.lenet5 import LeNet5
from pbt.database import ReadOnlyDatabase
from pbt.analyze import Analyzer 
from pbt.loss import CategoricalCrossEntropy, Accuracy, F1
from pbt.trainer import Trainer
from pbt.evaluator import Evaluator
from pbt.hyperparameters import Hyperparameters, ContiniousHyperparameter, DiscreteHyperparameter
from pbt.member import Checkpoint
from pbt.task.mnist import MnistKnowledgeSharing
from pbt.trainingservice import TrainingService

class TestTraining(unittest.TestCase):
    def setUp(self):
        population_size = 10
        batch_size = 64
        n_jobs = 7
        task = MnistKnowledgeSharing('lenet5')
        devices = ['cuda:0']
        loss_metric = 'cce'
        eval_metric = 'acc'
        loss_functions = {
            'cce': CategoricalCrossEntropy(),
            'acc': Accuracy()
        }
        trainer = Trainer(
            model_class=task.model_class,
            optimizer_class=task.optimizer_class,
            train_data=task.datasets.train,
            batch_size=batch_size,
            loss_functions=task.loss_functions,
            loss_metric=task.loss_metric)
        evaluator = Evaluator(
            model_class=task.model_class,
            test_data=task.datasets.eval,
            batch_size=batch_size,
            loss_group='eval',
            loss_functions=task.loss_functions)
        hp = Hyperparameters(
            optimizer={
                'lr': ContiniousHyperparameter(0.000001, 0.1, value=0.01),
                'momentum': ContiniousHyperparameter(0.001, 0.9, value=0.5)
            })
        self.checkpoints = [Checkpoint(
            id=i,
            parameters=hp,
            loss_metric=loss_metric,
            eval_metric=eval_metric,
            minimize=loss_functions[eval_metric].minimize)
            for i in range(population_size)]
        self.trainingservice = TrainingService(trainer=trainer, evaluator=evaluator, devices=devices, n_jobs=n_jobs, verbose=False)
        self.trainingservice.start()

    def tearDown(self):
        self.trainingservice.stop()
    
    def test_training_service(self):
        epochs = 5
        step_size = 10
        old_checkpoints = list(self.checkpoints)
        for epoch in range(epochs):
            new_checkpoints = list(self.trainingservice.train(candidates=old_checkpoints, train_step_size=step_size))
            for new_checkpoint in new_checkpoints:
                old_checkpoint = next(c for c in old_checkpoints if c.id == new_checkpoint.id)
                self.assertNotEqual(id(old_checkpoint), id(new_checkpoint))
                self.assertNotEqual(old_checkpoint, new_checkpoint)
                self.assertNotEqual(old_checkpoint.steps, new_checkpoint.steps)
                self.assertNotEqual(id(old_checkpoint.model_state), id(new_checkpoint.model_state))
                self.assertNotEqual(id(old_checkpoint.optimizer_state), id(new_checkpoint.optimizer_state))
                if epoch > 0:
                    for old_tensor, new_tensor in zip(
                        unwrap_iterable(old_checkpoint.model_state, exceptions=[torch.Tensor]),
                        unwrap_iterable(new_checkpoint.model_state, exceptions=[torch.Tensor])):
                        self.assertTrue(id(old_tensor) != id(new_tensor))
                    for old_tensor, new_tensor in zip(
                        unwrap_iterable(old_checkpoint.optimizer_state, exceptions=[torch.Tensor]),
                        unwrap_iterable(new_checkpoint.optimizer_state, exceptions=[torch.Tensor])):
                        self.assertTrue(id(old_tensor) != id(new_tensor))
            old_checkpoints = new_checkpoints