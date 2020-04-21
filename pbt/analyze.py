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

    def __create_score_dataframe(self):
        # create data holders
        loss_dataframe = pd.DataFrame()
        # aquire plot data
        for entry_id, entries in self.database.to_dict().items():
            for step, entry in entries.items():
                loss_dataframe.at[step, entry_id] = entry.test_score()
        return loss_dataframe

    def __create_loss_dataframes(self):
        # create data holders
        loss_dataframes = dict()
        # aquire plot data
        for entry_id, entries in self.database.to_dict().items():
            for step, entry in entries.items():
                for metric_type, metric_value in flatten_dict(entry.loss, delimiter ='_').items():
                    if metric_type not in loss_dataframes:
                        loss_dataframes[metric_type] = pd.DataFrame()
                    loss_dataframes[metric_type].at[step, entry_id] = metric_value
        return loss_dataframes

    def __create_time_dataframes(self):
        time_dataframes = dict()
        # aquire plot data
        for entry_id, entries in self.database.to_dict().items():
            for step, entry in entries.items():
                for time_group, time_value in flatten_dict(entry.time, delimiter ='_').items():
                    if time_group not in time_dataframes:
                        time_dataframes[time_group] = pd.DataFrame()
                    time_dataframes[time_group].at[step, entry_id] = time_value
        return time_dataframes

    def __create_hp_dataframes(self):
        hp_dataframes = dict()
        # aquire plot data
        for entry_id, entries in self.database.to_dict().items():
            for step, entry in entries.items():
                for hp_type, hp_value in entry.parameters.items(full_key = True):
                    if hp_type not in hp_dataframes:
                        hp_dataframes[hp_type] = pd.DataFrame()
                    hp_dataframes[hp_type].at[step, entry_id] = hp_value.value
        return hp_dataframes

    def create_loss_plot_files(self, save_directory):
        for metric_type, df in self.__create_loss_dataframes().items():
            if df.empty:
                continue
            # save dataframe to csv-file
            df.to_csv(Path(save_directory, f"loss_{metric_type}.csv"))
            # define plot 
            column_names = list(df.columns)
            ax = df.plot(title=metric_type, legend=False, subplots = False, sharex = True, figsize = (10,5), kind = 'line',  ls='none', marker='o', ms=4)
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
            # define plot
            ax = df.plot(title=time_group, legend=False, subplots = False, sharex = True, figsize = (10,5), kind = 'line',  ls='none', marker='o', ms=4)
            ax.set_xlabel('steps')
            ax.set_ylabel('seconds')
            # save plot to file
            plt.savefig(fname=Path(save_directory, f"time_{time_group}_plot.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"time_{time_group}_plot.svg"), format='svg', transparent=True)
            # save dataframe to csv-file
            df.to_csv(Path(save_directory, f"time_{time_group}.csv"))
            # clear plot
            plt.clf()

    def create_hyper_parameter_plot_files(self, save_directory, default_marker='', highlight_marker='s', default_marker_size = 4, highlight_marker_size = 4, default_alpha = 0.4):
        # set colors
        highlight_color = '#000000' # black
        default_color = '#C0C0C0' # silver
        # determine best member
        best = get_best_member(self.database)
        best_entry_id = f"{best.id:03d}"
        # get hyper-parameter dataframe
        hp_df = self.__create_hp_dataframes()
        # plot data
        for param_name, df in hp_df.items():
            # create dataframe views
            hp_df_without_best=df.drop(best_entry_id, axis=1)
            hp_df_with_best=df[best_entry_id]
            # plot lines
            ax = hp_df_without_best.plot(title=param_name, legend=False, subplots = False, sharex = True, figsize = (10,5), kind='line', ls='solid', alpha=default_alpha, marker=default_marker, ms=default_marker_size)
            # plot best
            ax = hp_df_with_best.plot(ax=ax, legend=False, kind='line', ls='solid', color=highlight_color, marker=highlight_marker, ms=highlight_marker_size)
            # set axis labels
            ax.set_xlabel('steps')
            ax.set_ylabel('value')
            # save figures to directory
            filename = f"hp_{param_name.replace('/', '_')}"
            plt.savefig(fname=Path(save_directory, f"{filename}.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"{filename}.svg"), format='svg', transparent=True)
            # save dataframe to csv-file
            df.to_csv(Path(save_directory, f"{filename}.csv"))
            plt.clf()