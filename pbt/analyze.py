import csv
import math
import random
import itertools
from pathlib import Path
from collections import defaultdict
from statistics import stdev, mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D

from .database import ReadOnlyDatabase
from .evaluator import Evaluator
from .hyperparameters import ContiniousHyperparameter, Hyperparameters
from .utils.constraint import clip
from .utils.iterable import flatten_dict

def ylim_outliers(data, minimum=None, maximum=None, strength = 1.5):
    if len(data) < 2 or len(set(data)) == 1:
        return
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
    # replace NaN values with None
    lower = None if math.isnan(lower) else lower
    upper = None if math.isnan(upper) else upper
    # limit
    plt.ylim(bottom=lower, top=upper, auto=False)

def get_best_member(database : ReadOnlyDatabase):
    best = None
    for member in database:
        best = member if best is None or member.steps > best.steps or (member.steps == best.steps and member >= best) else best
    return best

class Analyzer(object):
    def __init__(self, database : ReadOnlyDatabase, verbose : bool = False):
        self.database : ReadOnlyDatabase = database
        self.verbose = verbose

    def __print(self, message : str):
        if self.verbose:
            print(f"Analyzer: {message}")

    def test(self, evaluator : Evaluator, save_directory : str, device : str = 'cpu'):
        self.__print(f"Finding best member in population...")
        best = get_best_member(self.database)
        self.__print(f"Testing {best}...")
        evaluator(best, device)
        # save top members to file
        result = f"Best checkpoint: {best}: {best.performance_details()}"
        with Path(save_directory).open('a+') as f:
            f.write(f"{result}\n\n")
        self.__print(result)
        return best

    def test_generations(self, evaluator : Evaluator, save_directory : str, device : str = 'cpu', limit : int = 10):
        tested_subjects = list()
        subjects = sorted((entry for entry in self.database if entry.model_state is not None), reverse=True, key=lambda e: e.steps)[:limit]
        for index, entry in enumerate(subjects, start=1):
            if entry.model_state is None:
                self.__print(f"({index}/{len(subjects)}) Skipping {entry} due to missing model state.")
                continue
            self.__print(f"({index}/{len(subjects)}) Testing {entry}...")
            evaluator(entry, device)
            tested_subjects.append(entry)
            self.__print(entry.performance_details())
        # determine best checkpoint
        minimize = next(iter(self.database)).minimize
        sort_method = min if minimize else max
        best_checkpoint = sort_method(tested_subjects, key=lambda c: c.test_score())
        result = f"Best checkpoint: {best_checkpoint}: {best_checkpoint.performance_details()}"
        # save top members to file
        with Path(save_directory).open('a+') as f:
            f.write(f"{result}\n\n")
            for checkpoint in tested_subjects:
                f.write(str(checkpoint) + "\n")
        self.__print(result)
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

    def __create_loss_dataframes(self):
        population_entries = self.database.to_dict()
        # create data holders
        loss_dataframes = dict()
        # aquire plot data
        for entry_id, entries in population_entries.items():
            for step, entry in entries.items():
                for metric_type, metric_value in flatten_dict(entry.loss, delimiter ='_').items():
                    if metric_type not in loss_dataframes:
                        loss_dataframes[metric_type] = pd.DataFrame()
                    loss_dataframes[metric_type].at[step, entry_id] = metric_value
        return loss_dataframes

    def __create_time_dataframes(self):
        population_entries = self.database.to_dict()
        time_dataframes = dict()
        # aquire plot data
        for entry_id, entries in population_entries.items():
            for step, entry in entries.items():
                for time_group, time_value in flatten_dict(entry.time, delimiter ='_').items():
                    if time_group not in time_dataframes:
                        time_dataframes[time_group] = pd.DataFrame()
                    time_dataframes[time_group].at[step, entry_id] = time_value
        return time_dataframes

    def create_loss_plot_files(self, save_directory):
        for metric_type, df in self.__create_loss_dataframes().items():
            if df.empty:
                continue
            # save dataframe to csv-file
            df.to_csv(Path(save_directory, f"loss_{metric_type}.csv"))
            # define plot 
            column_names = list(df.columns)
            ax = df.reset_index().plot(x='index', y=column_names[1:], kind = 'line',  ls='none', marker='o', ms=4,
                title=metric_type, legend=False, subplots = False, sharex = True, figsize = (10,5))
            ax.set_xlabel('steps')
            ax.set_ylabel('loss')
            # save plot to file
            plt.savefig(fname=Path(save_directory, f"loss_{metric_type}.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"loss_{metric_type}.svg"), format='svg', transparent=True)
            # clear plot
            plt.clf()

    def create_time_plot_files(self, save_directory):
        for time_group, df in self.__create_time_dataframes().items():
            if df.empty:
                continue
            # save dataframe to csv-file
            df.to_csv(Path(save_directory, f"time_{time_group}.csv"))
            column_names = list(df.columns)
            # define plot
            ax = df.reset_index().plot(x='index', y=column_names[1:], kind = 'line',  ls='none', marker='o', ms=4,
                title=time_group, legend=False, subplots = False, sharex = True, figsize = (10,5))
            ax.set_xlabel('steps')
            ax.set_ylabel('seconds')
            # save plot to file
            plt.savefig(fname=Path(save_directory, f"time_{time_group}_plot.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"time_{time_group}_plot.svg"), format='svg', transparent=True)
            # clear plot
            plt.clf()

    def create_hyper_parameter_plot_files(self, save_directory, sensitivity=1, marker='o', min_marker_size = 4, max_marker_size = 8, cmap = "winter", best_color = "orange", worst_color = "red"):
        # get color map
        color_map = plt.get_cmap(cmap)
        tab_colors = [color_map(i/10) for i in range(0, 11, 1)]
        tab_map = matplotlib.colors.ListedColormap(tab_colors)
        # keep list of hyper_parameters
        hyper_parameters = set()
        # get entries
        population_entries = self.database.to_dict()
        best_entries = dict()
        worst_entries = dict()
        best_score = None
        worst_score = None
        # determine best and worst entries
        for entries in population_entries.values():
            for entry in entries.values():
                hyper_parameters.update(entry.parameters.keys())
                if entry.steps not in best_entries or entry > best_entries[entry.steps]:
                    best_entries[entry.steps] = entry
                    if not best_score or entry > best_score:
                        best_score = entry.score()
                if entry.steps not in worst_entries or entry < worst_entries[entry.steps]:
                    worst_entries[entry.steps] = entry
                    if not worst_score or (not math.isnan(entry.score()) and entry < worst_score):
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
                    scores[param_name].append(entry.parameters[param_name].normalized)
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
            y = [e.parameters[param_name].normalized for e in y]
            plt.scatter(x, y, label="worst score", color=worst_color, marker="o", s=6)
            # plot best score
            x, y = zip(*sorted(best_entries.items()))
            y = [e.parameters[param_name].normalized for e in y]
            plt.scatter(x, y, label="best score", color=best_color, marker="o", s=6)
            # legend
            plt.legend()
            # display colorbar
            plt.colorbar(hp_plot)
            # save figures to directory
            plt.savefig(fname=Path(save_directory, f"hp_plot_{param_name.replace('/', '_')}.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"hp_plot_{param_name.replace('/', '_')}.svg"), format='svg', transparent=True)
            plt.clf()