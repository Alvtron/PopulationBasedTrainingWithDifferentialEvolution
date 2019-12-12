import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
                    color = color_map(entry_index/(num_entries - 1))
                    point = (entry.steps, param.normalized())
                    ax.plot(point[0], point[1], marker='o', linestyle='-', linewidth=2, markersize=10, color=color)
                    if annotate: ax.annotate(f"{entry.score:.2f}", point)
            file_path_png = self.database.create_file_path(f"{entry_id:03d}_hyper_parameter_plot.png")
            file_path_svg = self.database.create_file_path(f"{entry_id:03d}_hyper_parameter_plot.svg")
            plt.savefig(fname=file_path_png, format='png', transparent=transparent)
            plt.savefig(fname=file_path_svg, format='svg', transparent=transparent)