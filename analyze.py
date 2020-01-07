import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from pathlib import Path
from database import Database
from evaluator import Evaluator
from hyperparameters import Hyperparameter, Hyperparameters
from utils import clip

class Analyzer(object):
    def __init__(self, database : Database, evaluator : Evaluator):
        self.database = database
        self.evaluator = evaluator

    def test(self, limit = None):
        entries = self.database.to_list()
        if limit:
            entries.sort(key=lambda e: e.eval_score, reverse=True)
            entries = entries[:limit]
        for entry in entries:
            entry.test_score = self.evaluator.eval(entry.model_state)
        return entries

    def create_plot_files(self, save_directory, n_hyper_parameters, min_score, max_score, annotate=False, sensitivity=1):
        color_map_key = "rainbow_r"
        color_map = plt.get_cmap(color_map_key)
        population_entries = self.database.to_dict()
        # set nubmer of rows and columns
        n_rows = round(math.sqrt(n_hyper_parameters))
        n_columns = math.ceil(n_hyper_parameters / n_rows)
        for entry_id, entries in population_entries.items(): # for each member
            # create figure and axes
            figure, axes = plt.subplots(n_rows, n_columns, sharex=True, figsize=(8,8))
            print(f"Creating plot for member {entry_id}...")
            for _, entry in entries.items(): # for each entry
                for param_index, (param_name, param) in enumerate(entry.hyper_parameters): # for each hyper-parameter
                    # prepare subplot
                    ax = axes.flat[param_index]
                    ax.set_title(param_name)
                    ax.set_ylim(bottom=0.0, top=1.0, auto=False)
                    ax.set(xlabel='steps', ylabel='value')
                    score_decimal = (entry.eval_score - min_score) / (max_score - min_score)
                    color = color_map(score_decimal ** sensitivity)
                    marker_size = clip(12 * score_decimal, 4, 12)
                    x, y = (entry.steps, param.normalized())
                    # plot
                    ax.plot(x, y, marker='o', markersize=marker_size, color=color)
                    if annotate: ax.annotate(f"{entry.eval_score:.2f}", (x, y))
            # delete not needed axes from the last row
            n_unused = len(axes.flat) - n_hyper_parameters
            if n_unused > 0:
                for ax in axes.flat[-n_unused:]:
                    ax.remove()
            # align y-labels, TODO: if y is normalized hyper-parameter
            figure.align_ylabels()
            # save figures to database directory
            plt.savefig(fname=Path(save_directory, f"{entry_id}_hyper_parameter_plot.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"{entry_id}_hyper_parameter_plot.svg"), format='svg', transparent=True)