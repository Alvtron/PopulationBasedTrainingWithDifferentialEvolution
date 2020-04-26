import csv
import math
import random
import itertools
from functools import partial
from pathlib import Path
from collections import defaultdict
from statistics import stdev, mean
from scipy import stats

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

def remove_outliers_from_dataframe(df, stdev_proximity = 3):
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < stdev_proximity).all(axis=1)
    return df[filtered_entries]

def ylim_from_df(df, strength = 1.5):
    if df.empty:
        return
    df = df.dropna(axis=1, inplace=False, how='any')
    # calculate statistics
    data_max = df.max().max()
    data_min = df.min().min()
    data_mean = df.values.mean()
    data_std = df.values.std(ddof=1)
    # identify outliers
    cut_off = data_std * strength
    lower = data_mean - cut_off
    upper = data_mean + cut_off
    if is_increasing(df):
        return lower, data_max + (data_max - lower) * 0.1
    else:
        return data_min + (data_min - upper) * 0.1, upper

def is_increasing(df, order=1):
    average_slope = 0.0
    for column_name in df:
        column = df[column_name].dropna(inplace=False, how='any')
        coeffs = np.polyfit(column.index.values, column, order)
        slope = coeffs[-2]
        average_slope += float(slope)
    return average_slope >= 0

def save_figure_to_files(figure, directory, filename : str):
    figure.savefig(fname=Path(directory, f"{filename}.png"), format='png', transparent=False)
    figure.savefig(fname=Path(directory, f"{filename}.svg"), format='svg', transparent=True)

class Analyzer(object):
    def __init__(self, database : ReadOnlyDatabase, verbose : bool = False):
        self.database : ReadOnlyDatabase = database
        self.verbose = verbose

    def __print(self, message : str):
        if self.verbose:
            print(f"Analyzer: {message}")

    def __minimize_score(self):
        for member in self.database:
            return member.minimize
        raise Exception

    def __get_best_member(self):
        best = None
        for member in self.database:
            best = member if best is None or member.steps > best.steps or (member.steps == best.steps and member >= best) else best
        return best

    def __get_worst_member(self):
        worst = None
        for member in self.database:
            worst = member if worst is None or member.steps < worst.steps or (member.steps == worst.steps and member <= worst) else worst
        return worst

    def test(self, evaluator : Evaluator, save_directory : str, device : str = 'cpu'):
        self.__print(f"Finding best member in population...")
        best = self.__get_best_member()
        self.__print(f"Testing {best}...")
        evaluator(best, device)
        # save top members to file
        result = f"Best checkpoint: {best}: {best.performance_details()}"
        with Path(save_directory).open('a+') as f:
            f.write(f"{result}\n\n")
        self.__print(result)
        return best

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
        score_dataframe = pd.DataFrame()
        # aquire plot data
        for entry_id, entries in self.database.to_dict().items():
            for step, entry in entries.items():
                score_dataframe.at[step, entry_id] = entry.test_score()
        score_dataframe.index.name = "steps"
        score_dataframe.sort_index(inplace=True)
        return score_dataframe

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
        for df in loss_dataframes.values():
            df.index.name = "steps"
            df.sort_index(inplace=True)
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
        for df in time_dataframes.values():
            df.index.name = "steps"
            df.sort_index(inplace=True)
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
        for df in hp_dataframes.values():
            df.index.name = "steps"
            df.sort_index(inplace=True)
        return hp_dataframes

    def create_loss_plot_files(self, save_directory):
        self.__create_loss_plot_files(directory=save_directory, prefix='loss', figsize=(10,7), save_csv=True)
        self.__create_loss_plot_files(directory=save_directory, prefix='loss', suffix='small', figsize=(10,4), save_csv=False)

    def __create_loss_plot_files(self, directory, prefix : str = None, suffix : str = None, figsize : tuple = (10,7), save_csv : bool = False):
        # default style
        DEFAULT_COLOR = 'darkcyan'
        DEFAULT_LINE_SIZE = 2
        # highlight style
        HIGHLIGHT_COLOR = 'darkcyan'
        HIGHLIGHT_LINE_SIZE = 4
        # nan style
        NAN_COLOR = 'crimson'
        NAN_MARKER_SIZE=6
        NAN_MARKER='x'
        # plot
        for metric_type, df in self.__create_loss_dataframes().items():
            if df.empty:
                continue
            # create figure
            fig, ax = plt.subplots(figsize = figsize, sharex = True)
            ax.set_title(metric_type)
            ax.set_xlabel("steps")
            ax.set_ylabel("value")
            ax.set_ylim(ylim_from_df(df, strength=2.0))
            # plot all
            df.plot(ax = ax, legend=False, kind = 'line',  ls='solid', color=DEFAULT_COLOR, linewidth=DEFAULT_LINE_SIZE, alpha=0.2)
            # plot best
            minimize = not is_increasing(df)
            best_member_id = df.tail(1).idxmin(axis=1).to_numpy()[0] if minimize else df.tail(1).idxmax(axis=1).to_numpy()[0] 
            df[best_member_id].plot(ax=ax, legend=True,  label='best', kind = 'line',  ls='solid', color=HIGHLIGHT_COLOR, linewidth=HIGHLIGHT_LINE_SIZE)
            ax.set_xlabel('steps')
            ax.set_ylabel('loss')
            # plot end points
            nan_df = pd.DataFrame()
            for column_name in df.loc[:, df.isna().any()]:
                index = df[column_name].last_valid_index()
                nan_df.at[index, 'end'] = df.at[index, column_name]
            if not nan_df.empty:
                nan_df.plot(ax=ax, legend=True, kind='line', ls='none', marker=NAN_MARKER, ms=NAN_MARKER_SIZE, color=NAN_COLOR)
            # draw grid
            ax.grid(axis='y', color='black', linestyle='-', linewidth=0.25)
            # save figures to directory
            filename = '_'.join(filter(None, [prefix, metric_type, suffix]))
            save_figure_to_files(fig, directory, filename)
            # save dataframe to csv-file
            if save_csv:
                df.to_csv(Path(directory, f"{filename}.csv"))
            # clear plot
            plt.clf()

    def create_time_plot_files(self, save_directory):
        self.__create_time_plot_files(directory=save_directory, prefix='time', figsize = (10,7), save_csv=True)
        self.__create_time_plot_files(directory=save_directory, prefix='time', suffix='small', figsize = (10,4), save_csv=False)

    def __create_time_plot_files(self, directory, prefix : str = None, suffix : str = None, figsize = (10,7), save_csv : bool = False):
        for time_group, df in self.__create_time_dataframes().items():
            if df.empty:
                continue
            # create figure
            fig, ax = plt.subplots(figsize = figsize, sharex = True)
            ax.set_title(time_group)
            ax.set_xlabel('steps')
            ax.set_ylabel('seconds')
            ax.set_ylim(ylim_from_df(df, strength=3.0))
            # plot
            df.plot(ax = ax, legend=False, kind = 'line',  ls='none', marker='o', ms=4)
            # draw grid
            ax.grid(axis='y', color='black', linestyle='-', linewidth=0.25)
            # save figures to directory
            filename = '_'.join(filter(None, [prefix, time_group, suffix]))
            save_figure_to_files(fig, directory, filename)
            # save dataframe to csv-file
            if save_csv:
                df.to_csv(Path(directory, f"{filename}.csv"))
            # clear plot
            plt.clf()

    def __create_color(self, score, worst_score, best_score, color_map, sensitivity = 20, epsilon = 1e-7):
        score_decimal = (score - worst_score + epsilon) / (best_score - worst_score + epsilon)
        adjusted_score_decimal = score_decimal# ** sensitivity
        color_value = color_map(adjusted_score_decimal)
        return color_value

    def create_hyper_parameter_plot_files(self, save_directory):
        self.__create_hyper_parameter_plot_files_v1(directory=save_directory, prefix='hp', suffix='dots', figsize=(10,7), save_csv=True)
        self.__create_hyper_parameter_plot_files_v2(directory=save_directory, prefix='hp', suffix='lines', figsize=(10,7), save_csv=False)
        self.__create_hyper_parameter_plot_files_v2(directory=save_directory, prefix='hp', suffix='lines_small', figsize=(10,4), save_csv=False)

    def __create_hyper_parameter_plot_files_v1(self, directory, prefix : str = '', suffix : str = '', figsize : tuple = (10,7), save_csv : bool = False):
        # set color map
        COLOR_MAP = plt.get_cmap('winter')
        TAB_COLORS = [COLOR_MAP(i/10) for i in range(0, 11, 1)]
        TAB_MAP = matplotlib.colors.ListedColormap(TAB_COLORS)
        # set colors
        DEFAULT_LINE_COLOR = '#C0C0C0' # gray
        WORST_COLOR = 'red'
        BEST_COLOR = 'orange'
        # set markers
        DEFAULT_MARKER_SIZE = 6
        BEST_MARKER_SIZE = 6
        WORST_MARKER_SIZE = 6
        DEFAULT_MARKER='s'
        BEST_MARKER='s'
        WORST_MARKER='s'
        # create score dataframe
        score_df = self.__create_score_dataframe()
        # create color dataframe
        if self.__minimize_score():
            best_member_id = score_df.tail(1).idxmin(axis=1).to_numpy()[0]
            worst_member_id = score_df.tail(1).idxmax(axis=1).to_numpy()[0]
            best_score = score_df.min().min()
            worst_score = score_df.max().max()
        else:
            best_member_id = score_df.tail(1).idxmax(axis=1).to_numpy()[0]
            worst_member_id = score_df.tail(1).idxmin(axis=1).to_numpy()[0]
            worst_score = score_df.max().max()
            best_score = score_df.min().min()
        color_df = score_df.applymap(partial(self.__create_color, worst_score=worst_score, best_score=best_score, color_map=TAB_MAP, sensitivity=10))
        # create hyper-parameter dataframe
        hp_df = self.__create_hp_dataframes()
        # plot data
        for param_name, df in hp_df.items():
            # create figure
            fig, ax = plt.subplots(figsize = figsize, sharex = True)
            ax.set_title(param_name)
            ax.set_xlabel("steps")
            ax.set_ylabel("value")
            # plot default
            for column_name in df:
                df.plot(y=column_name, ax=ax, legend=False, kind='line', ls='none', color=color_df[column_name], marker=DEFAULT_MARKER, ms=DEFAULT_MARKER_SIZE, label=column_name)
            # plot colorbar
            sm = plt.cm.ScalarMappable(cmap=TAB_MAP, norm=plt.Normalize(vmin=0.0, vmax=1.0))
            colorbar = fig.colorbar(sm, ax=ax)
            colorbar.set_label('performance')
            # plot worst
            hp_df_with_worst=df[worst_member_id]
            hp_df_with_worst.plot(ax=ax, legend=True, kind='line', ls='none', marker=WORST_MARKER, ms=WORST_MARKER_SIZE, color=WORST_COLOR, label='worst')
            # plot best
            hp_df_with_best=df[best_member_id]
            hp_df_with_best.plot(ax=ax, legend=True, kind='line', ls='none', marker=BEST_MARKER, ms=BEST_MARKER_SIZE, color=BEST_COLOR, label='best')
            # save figures to directory
            param_formatted = param_name.replace('/', '_')
            filename = '_'.join(filter(None, [prefix, param_formatted, suffix]))
            save_figure_to_files(fig, directory, filename)
            # save dataframe to csv-file
            if save_csv:
                df.to_csv(Path(directory, f"{filename}.csv"))
            plt.clf()
        
    def __create_hyper_parameter_plot_files_v2(self, directory, prefix : str = None, suffix : str = None, figsize : tuple = (10,2), save_csv : bool = False):
        # set colors
        FILL_COLOR = 'gainsboro'
        MEAN_COLOR = 'gray'
        LINE_COLOR = 'cadetblue'
        HIGHLIGHT_COLOR = 'darkcyan'
        NAN_COLOR = 'crimson'
        # set markers
        DEFAULT_LINE_SIZE = 2
        HIGHLIGHT_LINE_SIZE = 2
        NAN_MARKER_SIZE=6
        HIGHLIGHT_MARKER='s'
        NAN_MARKER='x'
        # get best and worst members
        score_df = self.__create_score_dataframe()
        if self.__minimize_score():
            best_member_id = score_df.tail(1).idxmin(axis=1).to_numpy()[0]
            worst_member_id = score_df.tail(1).idxmax(axis=1).to_numpy()[0]
        else:
            best_member_id = score_df.tail(1).idxmax(axis=1).to_numpy()[0]
            worst_member_id = score_df.tail(1).idxmin(axis=1).to_numpy()[0]
        # create hyper-parameter dataframe
        hp_df = self.__create_hp_dataframes()
        # plot data
        for param_name, df in hp_df.items():
            # create figure
            fig, ax = plt.subplots(figsize = figsize, sharex = True)
            ax.set_title(param_name)
            ax.set_xlabel("steps")
            ax.set_ylabel("value")
            # plot fill
            ax.fill_between(df.index, y1=df.min(axis=1), y2=df.max(axis=1), color=FILL_COLOR)
            # plot lines
            df.plot(ax=ax, legend=False, kind='line', ls='solid', solid_capstyle='round', linewidth=DEFAULT_LINE_SIZE, color=HIGHLIGHT_COLOR, alpha=0.2)
            df.mean(axis=1).plot(ax=ax, legend=True, kind='line', ls='solid', linewidth=DEFAULT_LINE_SIZE, color=MEAN_COLOR, label='mean')
            # plot best
            hp_df_with_best=df[best_member_id]
            hp_df_with_best.plot(ax=ax, legend=True, kind='line', ls='solid', linewidth=HIGHLIGHT_LINE_SIZE, color=HIGHLIGHT_COLOR, marker=HIGHLIGHT_MARKER, ms=6, label='best')
            # plot end points
            nan_df = pd.DataFrame()
            for column_name in df.loc[:, df.isna().any()]:
                index = df[column_name].last_valid_index()
                nan_df.at[index, 'end'] = df.at[index,column_name]
            if not nan_df.empty:
                nan_df.plot(ax=ax, legend=True, kind='line', ls='none', marker=NAN_MARKER, ms=NAN_MARKER_SIZE, color=NAN_COLOR)
            # save figures to directory
            param_formatted = param_name.replace('/', '_')
            filename = '_'.join(filter(None, [prefix, param_formatted, suffix]))
            save_figure_to_files(fig, directory, filename)
            # save dataframe to csv-file
            if save_csv:
                df.to_csv(Path(directory, f"{filename}.csv"))
            plt.clf()