import unittest
import shutil
import itertools

import torch

from pbt.database import ReadOnlyDatabase

DATABASE_PATH = "path/to/database/"

class TestDatabase(unittest.TestCase):
    
    def setUp(self):
        self.database = ReadOnlyDatabase(DATABASE_PATH, read_function=torch.load)

    def test_database(self):
        self.assertTrue(self.database.exists)
        len(self.database)
        entry = self.database.entry(0, 250)
        self.assertEqual(entry.uid, 0)
        self.assertEqual(entry.steps, 250)
        entry = self.database.entry(0, 300)
        self.assertTrue(entry is None)
        self.assertTrue(0 in self.database)
        self.assertFalse(50 in self.database)
        list(self.database.entries(0))
        list(itertools.islice(self.database, 0, 10))