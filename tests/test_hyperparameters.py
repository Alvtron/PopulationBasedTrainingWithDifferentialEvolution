import unittest
import random
import copy

from pbt.hyperparameters import ContiniousHyperparameter, DiscreteHyperparameter, Hyperparameters
from pbt.utils.iterable import unwrap_iterable

class TestHyperparameters(unittest.TestCase):

    def test_initialization(self):
        a = ContiniousHyperparameter(0, 100, value=25)
        b = ContiniousHyperparameter(0, 100, value=25)
        self.assertEqual(a, b)
        a.value = 50
        self.assertNotEqual(a, b)
        a.normalized = 0.5
        self.assertEqual(a.value, 50)
        a.normalized = 0.75
        self.assertEqual(a.value, 75)
        a.value = 130
        self.assertEqual(a.value, 100)
        a.value = -0.40
        self.assertEqual(a.value, 0)

    def test_math(self):
        a = ContiniousHyperparameter(0, 100, value=50)
        b = ContiniousHyperparameter(0, 100, value=25)
        c = a + b
        self.assertIsInstance(c, ContiniousHyperparameter)
        self.assertEqual(a.value, 50)
        self.assertEqual(b.value, 25)
        self.assertEqual(c.value, 75)
        i = a + 0.1
        self.assertEqual(i.value, 60)
        d = a - b
        e = b - a
        self.assertEqual(d.value, 25)
        self.assertEqual(e.value, 0)
        j = a - 0.1
        self.assertEqual(j.value, 40)
        f = b * 2
        self.assertEqual(f.value, 50)
        g = a / 2
        self.assertEqual(g.value, 25)
        h = b ** -1
        self.assertEqual(h.value, 1/25)

    def test_categorical_hyperparameters(self):
        c_hp = DiscreteHyperparameter('A','B','C','D','E','F','G','H','I', 'J', 'K', value='B')
        self.assertEqual(c_hp.value, 'B')
        self.assertEqual(c_hp.normalized, 0.10)
        self.assertEqual(c_hp.lower_bound, 0)
        self.assertEqual(c_hp.upper_bound, 10)
        self.assertEqual(c_hp.from_normalized(0.30), 'D')
        self.assertEqual(c_hp.from_normalized(0.26), 'D')
        self.assertEqual(c_hp.from_normalized(0.34), 'D')
        for i, value in enumerate(c_hp.search_space):
            self.assertEqual(c_hp.from_value(value), i/10)

    def test_hyperparameter_configuration(self):
        a = Hyperparameters(
            augment = {
                'a': ContiniousHyperparameter(1, 256, value=100)},
            model = {
                'b': ContiniousHyperparameter(1e-6, 1e-0, value=1e-1)},
            optimizer = {
                'c': ContiniousHyperparameter(0.0, 1e-5, value=1e-6),
                'd': DiscreteHyperparameter(False, True, value=False)
                })
        b = Hyperparameters(
            augment = {
                'a': ContiniousHyperparameter(1, 256, value=50)},
            model = {
                'b': ContiniousHyperparameter(1e-6, 1e-0, value=1e-2)},
            optimizer = {
                'c': ContiniousHyperparameter(0.0, 1e-5, value=1e-6),
                'd': DiscreteHyperparameter(False, True, value=True)
                })
        self.assertEqual(len(a), 4)
        self.assertNotEqual(a, b)
        for a_hp, b_hp in zip(a, b):
            a_hp.value = b_hp.value
        self.assertEqual(a, b)
        c = copy.deepcopy(a)
        _p = ContiniousHyperparameter(1e-6, 1e-0, value=1.0)
        c[1] = _p
        self.assertEqual(c[1], _p)
        self.assertNotEqual(c[1], a[1])
        self.assertNotEqual(c, a)