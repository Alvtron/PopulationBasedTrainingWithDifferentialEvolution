import math
import matplotlib.pyplot as plt
import random
from database import Checkpoint
from hyperparameters import Hyperparameter, Hyperparameters

class Analyzer(object):
    def __init__(self, database, evaluator):
        self.database = database
        self.evaluator = evaluator

    def test(self, limit = None):
        entries = self.database.to_list()
        if limit:
            entries.sort(key=lambda e: e.score, reverse=True)
            entries = entries[:limit]
        for entry in entries:
            entry.score = self.evaluator.eval(entry.model_state)
        return entries

    def plot_hyperparams(self, member_id, nrows = 3):
        population_entries = self.database.to_dict()
        entries = population_entries[member_id]
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        for steps, entry in entries.items():
            ncolumns = math.ceil(len(entry.hyper_parameters) / nrows)
            for param_index, (param_name, param) in enumerate(entry.hyper_parameters):
                ax = plt.subplot(nrows, ncolumns, param_index + 1)
                ax.set_title(param_name)
                ax.scatter(param.value(), entry.score * 0.01)
        plt.show()