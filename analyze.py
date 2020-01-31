import math
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import random
import pickle
import itertools
import numpy as np
from pathlib import Path
from database import ReadOnlyDatabase
from evaluator import Evaluator
from hyperparameters import Hyperparameter, Hyperparameters
from utils.constraint import clip
from utils.iterable import flatten_dict
from collections import defaultdict
from statistics import stdev, mean

def ylim_outliers(data, minimum=None, maximum=None, strength = 1.5):
    # calculate summary statistics
    data_mean = mean(data)
    data_std = stdev(data)
    # identify outliers
    cut_off = data_std * strength
    lower = data_mean - cut_off
    upper = data_mean + cut_off
    # constrain
    lower = max(lower, minimum) if minimum else lower
    upper = min(upper, maximum) if maximum else upper
    # limit
    plt.ylim(bottom=lower, top=upper, auto=False)

class Analyzer(object):
    def __init__(self, database : ReadOnlyDatabase):
        self.database : ReadOnlyDatabase = database

    def test(self, evaluator : Evaluator, save_directory, limit = None, verbose = False):
        tested_subjects = list()
        minimize = next(iter(self.database)).minimize
        subjects = sorted(self.database, reverse=True)[:limit]
        for index, entry in enumerate(subjects, start=1):
            if not entry.has_model_state():
                if verbose: print(f"({index}/{len(subjects)}) Skipping {entry} due to missing model state.")
                continue
            model_state, _ = entry.load_state()
            if verbose: print(f"({index}/{len(subjects)}) Testing {entry}...", end=" ")
            entry.loss['test'] = evaluator.eval(model_state)
            tested_subjects.append(entry)
            if verbose:
                for metric_type, metric_value in entry.loss['test'].items():
                    print(f"{metric_type}: {metric_value:4f}", end=" ")
                print(end="\n")
        # determine best checkpoint
        sort_method = min if minimize else max
        best_checkpoint = sort_method(tested_subjects, key=lambda c: c.test_score())
        result = f"Best checkpoint: {best_checkpoint}: {best_checkpoint.performance_details()}"
        # save top members to file
        with Path(save_directory).open('a+') as f:
            f.write(f"{result}\n\n")
            for checkpoint in tested_subjects:
                f.write(str(checkpoint) + "\n")
        if verbose: print(result)
        return tested_subjects

    def create_statistics(self, save_directory):
        population_entries = self.database.to_dict()
        # get member statistics
        checkpoint_summaries = dict()
        for entry_id, entries in population_entries.items():
            entries = entries.values()
            checkpoint_summaries[entry_id] = dict()
            summary = checkpoint_summaries[entry_id]
            summary['num_entries'] = len(entries)
            for checkpoint in entries:
                for time_type, time_value in checkpoint.time.items():
                    max_key = f"time_{time_type}_max"
                    min_key = f"time_{time_type}_min"
                    avg_key = f"time_{time_type}_avg"
                    total_key = f"time_{time_type}_total"
                    if max_key not in summary or time_value > summary[max_key]:
                        summary[max_key] = time_value
                    if min_key not in summary or time_value < summary[min_key]:
                        summary[min_key] = time_value
                    if avg_key not in summary:
                        summary[avg_key] = time_value / summary['num_entries']
                    else:
                        summary[avg_key] += time_value / summary['num_entries']
                    if total_key not in summary:
                        summary[total_key] = time_value
                    else:
                        summary[total_key] += time_value
                for loss_group, loss_values in checkpoint.loss.items():
                    for loss_type, loss_value in loss_values.items():
                        max_key = f"loss_{loss_group}_{loss_type}_max"
                        min_key = f"loss_{loss_group}_{loss_type}_min"
                        avg_key = f"loss_{loss_group}_{loss_type}_avg"
                        if max_key not in summary or loss_value > summary[max_key]:
                            summary[max_key] = loss_value
                        if min_key not in summary or loss_value < summary[min_key]:
                            summary[min_key] = loss_value
                        if avg_key not in summary:
                            summary[avg_key] = loss_value / summary['num_entries']
                        else:
                            summary[avg_key] += loss_value / summary['num_entries']
        # save/print member statistics
        for entry_id, checkpoint_summary in checkpoint_summaries.items():
            with open(f"{save_directory}/{entry_id}_statistics.txt", "a+") as file:
                for tag, statistic in checkpoint_summary.items():
                    info = f"{tag}: {statistic}"
                    file.write(info + "\n")

    def create_plot_files(self, save_directory):
        population_entries = self.database.to_dict()
        # create data holders
        loss_dict = defaultdict(lambda: defaultdict(dict))
        time_dict = defaultdict(lambda: defaultdict(dict))
        # aquire plot data
        for entry_id, entries in population_entries.items():
            for step, entry in entries.items():
                for metric_type, metric_value in flatten_dict(entry.loss, delimiter ='_').items():
                    if step not in loss_dict[metric_type][entry_id]:
                        loss_dict[metric_type][entry_id][step] = list()
                    loss_dict[metric_type][entry_id][step].append(metric_value)
                for time_group, time_value in flatten_dict(entry.time, delimiter ='_').items():
                    if step not in time_dict[time_group][entry_id]:
                        time_dict[time_group][entry_id][step] = list()
                    time_dict[time_group][entry_id][step].append(time_value)
        # plot loss
        for metric_type, members_entries in loss_dict.items():
            plt.xlabel("steps")
            plt.ylabel("value")
            plt.title(metric_type)
            data = list()
            for id, steps_entries in members_entries.items():
                sorted_entries = list(zip(*sorted(steps_entries.items())))
                data.extend(list(itertools.chain(*sorted_entries[1])))
                plt.scatter(sorted_entries[0], sorted_entries[1], marker="o", s=4, label=f"m_{id}")
            if 'acc' in metric_type:
                ylim_outliers(data, minimum=-5, maximum=105)
            else:
                ylim_outliers(data, minimum=-0.05, maximum=1.05)
            plt.savefig(fname=Path(save_directory, f"loss_{metric_type}_plot.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"loss_{metric_type}_plot.svg"), format='svg', transparent=True)
            plt.clf()
        # plot time
        for time_group, members_entries in time_dict.items():
            plt.xlabel("steps")
            plt.ylabel("seconds")
            plt.title(time_group)
            data = list()
            for id, steps_entries in members_entries.items():
                sorted_entries = list(zip(*sorted(steps_entries.items())))
                data.extend(list(itertools.chain(*sorted_entries[1])))
                plt.scatter(sorted_entries[0], sorted_entries[1], marker="o", s=4, label=f"m_{id}")
            ylim_outliers(data)
            plt.savefig(fname=Path(save_directory, f"time_{time_group}_plot.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"time_{time_group}_plot.svg"), format='svg', transparent=True)
            plt.clf()

    def create_hyper_parameter_plot_files(self, save_directory, sensitivity=1, marker='o', min_marker_size = 4, max_marker_size = 8, cmap = "winter", best_color = "orange", worst_color = "red"):
        # get color map
        color_map = plt.get_cmap(cmap)
        tab_colors = [color_map(i/10) for i in range(0, 11, 1)]
        tab_map = matplotlib.colors.ListedColormap(tab_colors)
        # get objective data
        objective_info = pickle.load(Path(self.database.path, "info", "parameters.obj").open("rb"))
        hyper_parameters = objective_info['hyper_parameters']
        # get entries
        population_entries = self.database.to_dict()
        best_entries = dict()
        worst_entries = dict()
        best_score = None
        worst_score = None
        # determine best and worst entries
        for entries in population_entries.values():
            for entry in entries.values():
                if entry.steps not in best_entries or entry > best_entries[entry.steps]:
                    best_entries[entry.steps] = entry
                    if not best_score or entry > best_score:
                        best_score = entry.score()
                if entry.steps not in worst_entries or entry < worst_entries[entry.steps]:
                    worst_entries[entry.steps] = entry
                    if not worst_score or entry < worst_score:
                        worst_score = entry.score()
        # create data holders
        steps = defaultdict(list)
        scores = defaultdict(list)
        colors = defaultdict(list)
        # aquire plot data
        for entries in population_entries.values():
            for entry in entries.values():
                score_decimal = (entry.score() - worst_score + 1e-7) / (best_score - worst_score + 1e-7)
                color = score_decimal ** sensitivity
                for param_name in hyper_parameters:
                    steps[param_name].append(entry.steps)
                    scores[param_name].append(entry.hyper_parameters[param_name].normalized)
                    colors[param_name].append(color)
        # plot data
        for param_name in hyper_parameters:
            # create sub-plots and axes
            plt.title(param_name)
            plt.ylim(bottom=0.0, top=1.0, auto=False)
            plt.xlabel("steps")
            plt.ylabel("value")
            # plot hyper_parameters
            hp_plot = plt.scatter(
                x=steps[param_name],
                y=scores[param_name],
                marker="s",
                s=6,
                c=colors[param_name],
                cmap=tab_map)
            # plot worst score
            x, y = zip(*sorted(worst_entries.items()))
            y = [e.hyper_parameters[param_name].normalized for e in y]
            plt.scatter(x, y, label="worst score", color=worst_color, marker="o", s=6)
            # plot best score
            x, y = zip(*sorted(best_entries.items()))
            y = [e.hyper_parameters[param_name].normalized for e in y]
            plt.scatter(x, y, label="best score", color=best_color, marker="o", s=6)
            # legend
            plt.legend()
            # display colorbar
            plt.colorbar(hp_plot)
            # save figures to directory
            plt.savefig(fname=Path(save_directory, f"hp_plot_{param_name.replace('/', '_')}.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"hp_plot_{param_name.replace('/', '_')}.svg"), format='svg', transparent=True)
            plt.clf()