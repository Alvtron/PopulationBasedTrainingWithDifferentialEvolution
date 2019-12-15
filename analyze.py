import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from database import Checkpoint
from utils import translate
from hyperparameters import Hyperparameter, Hyperparameters

class Analyzer(object):
    def __init__(self, database, evaluator):
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

    def create_plot_files(self, nrows=3, annotate=False, transparent=False):
        figure = plt.figure(figsize=(8,8))
        figure.tight_layout(pad=5.0)
        color_map_key = "winter"
        color_map = plt.get_cmap(color_map_key)
        population_entries = self.database.to_dict()
        for entry_id, entries in population_entries.items():
            num_entries = len(entries)
            print(f"Creating plot for member {entry_id}...")
            for entry_index, (_, entry) in enumerate(entries.items()):
                ncolumns = math.ceil(len(entry.hyper_parameters) / nrows)
                for param_index, (param_name, param) in enumerate(entry.hyper_parameters):
                    ax = figure.add_subplot(nrows, ncolumns, param_index + 1)
                    ax.set_title(param_name)
                    ax.set_xlabel('steps')
                    ax.set_ylabel('value')
                    ax.set_ylim(bottom=0.0, top=1.0, auto=False)
                    color = color_map(entry.eval_score/100)
                    point = (entry.steps, param.normalized())
                    ax.plot(point[0], point[1], marker='o', linestyle='-', linewidth=2, markersize=10, color=color)
                    if annotate: ax.annotate(f"{entry.eval_score:.2f}", point)
            figure.align_ylabels()
            file_path_png = self.database.create_file_path(f"{entry_id:03d}_hyper_parameter_plot.png")
            file_path_svg = self.database.create_file_path(f"{entry_id:03d}_hyper_parameter_plot.svg")
            plt.savefig(fname=file_path_png, format='png', transparent=transparent)
            plt.savefig(fname=file_path_svg, format='svg', transparent=transparent)