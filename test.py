import unittest

if __name__ == '__main__':
    from tests.test_hyperparameters import TestHyperparameters
    from tests.test_database import TestDatabase
    from tests.test_dataset import TestDataset
    print("unit tests running...")
    unittest.main()
    print("unit tests completed.")