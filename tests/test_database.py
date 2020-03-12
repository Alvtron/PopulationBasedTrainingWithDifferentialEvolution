import unittest
import shutil
import itertools

import torch

from pbt.database import ReadOnlyDatabase

class TestDatabase(unittest.TestCase):
    def test_database(self):
        database = ReadOnlyDatabase("tests/checkpoint/20200311045310", read_function=torch.load)
        self.assertTrue(database.exists)
        len(database)
        entry = database.entry(0, 250)
        self.assertEqual(entry.id, 0)
        self.assertEqual(entry.steps, 250)
        entry = database.entry(0, 300)
        self.assertTrue(entry is None)
        self.assertTrue(0 in database)
        self.assertFalse(50 in database)
        list(database.entries(0))
        list(itertools.islice(database, 0, 10))