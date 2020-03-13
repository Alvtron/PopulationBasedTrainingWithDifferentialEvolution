import unittest
import random
import time
from collections import defaultdict, Counter

import torch
import torchvision
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt

import pbt.utils.data
from pbt.models.lenet5 import LeNet5
from pbt.database import ReadOnlyDatabase
from pbt.analyze import Analyzer 
from pbt.loss import CategoricalCrossEntropy, Accuracy, F1
from pbt.utils.data import stratified_split, split, random_split

class TestDataset(unittest.TestCase):

    def setUp(self):
        train_data_path = test_data_path = './data'
        self.train_data = MNIST(
            train_data_path,
            train=True,
            download=True)
        self.test_data = MNIST(
            test_data_path,
            train=False,
            download=True)

    def test_dataset_split(self):
        test_data = self.test_data
        train_data, eval_data = split(self.train_data, fraction=50000/60000)
        self.assertEqual(len(self.train_data), len(train_data) + len(test_data))
        self.assertNotEqual(len(train_data), len(eval_data))
        self.assertNotEqual(len(train_data), len(test_data))
        self.assertEqual(len(eval_data), len(test_data))
        self.assertTrue(not any(y in train_data.indices for y in eval_data.indices))

    def test_dataset_random_split(self):
        test_data = self.test_data
        train_data, eval_data = random_split(self.train_data, fraction=50000/60000)
        self.assertEqual(len(self.train_data), len(train_data) + len(test_data))
        self.assertNotEqual(len(train_data), len(eval_data))
        self.assertNotEqual(len(train_data), len(test_data))
        self.assertEqual(len(eval_data), len(test_data))
        self.assertTrue(not any(y in train_data.indices for y in eval_data.indices))

    def test_stratified_split(self):
        fraction = 50000/60000
        test_data = self.test_data
        train_data, train_labels, eval_data, eval_labels = stratified_split(
            self.train_data, labels=self.train_data.targets, fraction=fraction, return_labels=True)
        self.assertEqual(len(self.train_data), len(train_data) + len(test_data))
        self.assertNotEqual(len(train_data), len(eval_data))
        self.assertNotEqual(len(train_data), len(test_data))
        self.assertEqual(len(eval_data), len(test_data))
        self.assertNotEqual(train_labels, eval_labels)
        self.assertTrue(not any(y in train_data.indices for y in eval_data.indices))
        train_label_counter = Counter(train_labels)
        eval_label_counter = Counter(eval_labels)
        self.assertNotEqual(train_label_counter, eval_label_counter)
        for label in train_label_counter:
            self.assertAlmostEqual(train_label_counter[label] + eval_label_counter[label], train_label_counter[label] / (fraction), delta=8)
            self.assertAlmostEqual(train_label_counter[label] + eval_label_counter[label], eval_label_counter[label] / (1.0 - fraction), delta=8)