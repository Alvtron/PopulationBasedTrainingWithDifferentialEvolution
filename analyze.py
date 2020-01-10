import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pickle
from pathlib import Path
from database import ReadOnlyDatabase
from evaluator import Evaluator
from hyperparameters import Hyperparameter, Hyperparameters
from utils.math import clip
from utils.iterable import flatten_dict

class Analyzer(object):
    def __init__(self, database : ReadOnlyDatabase):
        self.database = database

    def create_progression_dict(self):
        attributes = {'steps','epochs','hyper_parameters','eval_metric','loss','time'}
        population_entries = self.database.to_dict()
        checkpoint_progression = dict()
        for entry_id, entries in population_entries.items():
            checkpoint_progression[entry_id] = dict()
            for entry in entries.values():
                entry_dict = { attribute: entry.__dict__[attribute] for attribute in attributes }
                entry_dict['score'] = entry.score
                entry_dict = flatten_dict(entry_dict, exclude=['hyper_parameters'], delimiter='_')
                for attribute, value in entry_dict.items():
                    if not attribute in checkpoint_progression[entry_id]:
                        checkpoint_progression[entry_id][attribute] = list()
                    checkpoint_progression[entry_id][attribute] += [value]
        return checkpoint_progression

    def test(self, evaluator : Evaluator, limit = None):
        entries = self.database.to_list()
        if limit:
            entries.sort(key=lambda e: e.score, reverse=True)
            entries = entries[:limit]
        for entry in entries:
            entry.loss['test'] = evaluator.eval(entry.model_state)
        return entries

    def create_statistics(self, save_directory, verbose=False):
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
                    if min_key not in summary or time_value > summary[min_key]:
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
                        if min_key not in summary or loss_value > summary[min_key]:
                            summary[min_key] = loss_value
                        if avg_key not in summary:
                            summary[avg_key] = loss_value / summary['num_entries']
                        else:
                            summary[avg_key] += loss_value / summary['num_entries']
        # save/print member statistics
        for entry_id, checkpoint_summary in checkpoint_summaries.items():
            if verbose: print(f"Statistics for member {entry_id}:")
            with open(f"{save_directory}/{entry_id}_statistics.txt", "a+") as file:
                for tag, statistic in checkpoint_summary.items():
                    info = f"{tag}:{statistic}"
                    if verbose: print(info)
                    file.write(info + "\n")

    def create_plot_files(self, save_directory):
        exclude_attributes = ['steps','epochs','hyper_parameters','eval_metric','hyper_parameters','score']
        progression_dict = self.create_progression_dict()
        attributes = next(iter(progression_dict.values())).keys()
        for attribute in attributes:
            if attribute in exclude_attributes:
                continue
            plt.xlabel("steps")
            plt.ylabel("value")
            plt.title(attribute)
            for id in progression_dict:
                plt.plot(progression_dict[id][attribute], label=f"m_{id}")
            plt.legend()
            plt.savefig(fname=Path(save_directory, f"{attribute}_plot.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"{attribute}_plot.svg"), format='svg', transparent=True)
            plt.clf()

    def create_hyper_parameter_single_plot_files(self, save_directory, min_score, max_score, annotate=False, sensitivity=1, marker='o', min_marker_size = 4, max_marker_size = 10):
        # get color map
        color_map_key = "rainbow_r"
        color_map = plt.get_cmap(color_map_key)
        # get population data
        progression_dict = self.create_progression_dict()
        # get objective data
        objective_info = pickle.load(Path(self.database.path, "info", "parameters.obj").open("rb"))
        hyper_parameters = objective_info['hyper_parameters']
        for param_name in hyper_parameters:
            figure = plt.figure()
            plt.title(param_name)
            plt.ylim(bottom=0.0, top=1.0, auto=False)
            plt.xlabel('steps')
            plt.ylabel('value')
            for id in progression_dict:
                steps = [step for step in progression_dict[id]['steps']]
                parameter_values = [hp[param_name].normalized for hp in progression_dict[id]['hyper_parameters']]
                scores = [score for score in progression_dict[id]['score']]
                # plot markers first
                for step, parameter_value, score in zip(steps, parameter_values, scores):
                    score_decimal = (score - min_score) / (max_score - min_score)
                    color = color_map(score_decimal ** sensitivity)
                    marker_size = clip(max_marker_size * score_decimal, min_marker_size, max_marker_size)
                    # plot
                    plt.plot(step, parameter_value, marker, markersize=marker_size, color=color)
                    if annotate: plt.annotate(f"{score:.2f}", (step, parameter_value))
                # plot lines last
                plt.plot(steps, parameter_values, label=f"m_{id}")
            # legend
            figure.legend()
            # save figures to directory
            plt.savefig(fname=Path(save_directory, f"{param_name.replace('/', '_')}_plot.png"), format='png', transparent=False)
            plt.savefig(fname=Path(save_directory, f"{param_name.replace('/', '_')}_plot.svg"), format='svg', transparent=True)
            # clear current figure and axes
            plt.clf()

    def create_hyper_parameter_multi_plot_files(self, save_directory, min_score, max_score, annotate=False, sensitivity=1, marker='o', min_marker_size = 4, max_marker_size = 10):
        # get objective data
        objective_info = pickle.load(Path(self.database.path, "info", "parameters.obj").open("rb"))
        population_size = objective_info['population_size']
        hyper_parameters = objective_info['hyper_parameters']
        n_hyper_parameters = len(hyper_parameters)
        # get population data
        progression_dict = self.create_progression_dict()
        # get color map
        color_map_key = "rainbow_r"
        color_map = plt.get_cmap(color_map_key)
        # set number of rows and columns
        n_rows = round(math.sqrt(n_hyper_parameters))
        n_columns = math.ceil(n_hyper_parameters / n_rows)
        # create sub-plots and axes
        figure, axes = plt.subplots(n_rows, n_columns, sharex=True, figsize=(10,10))
        for param_index, param_name in enumerate(hyper_parameters):
            ax = axes.flat[param_index]
            ax.set_title(param_name)
            ax.set_ylim(bottom=0.0, top=1.0, auto=False)
            ax.set(xlabel='steps', ylabel='value')
            for id in progression_dict:
                steps = [step for step in progression_dict[id]['steps']]
                parameter_values = [hp[param_name].normalized for hp in progression_dict[id]['hyper_parameters']]
                scores = [score for score in progression_dict[id]['score']]
                # plot markers first
                for step, parameter_value, score in zip(steps, parameter_values, scores):
                    score_decimal = (score - min_score) / (max_score - min_score)
                    color = color_map(score_decimal ** sensitivity)
                    marker_size = clip(max_marker_size * score_decimal, min_marker_size, max_marker_size)
                    # plot
                    ax.plot(step, parameter_value, marker, markersize=marker_size, color=color)
                    if annotate: ax.annotate(f"{score:.2f}", (step, parameter_value))
                # plot lines last
                ax.plot(steps, parameter_values, label=f"m_{id}")
        # delete unused axes
        n_unused = len(axes.flat) - n_hyper_parameters
        if n_unused > 0:
            for ax in axes.flat[-n_unused:]:
                ax.remove()
        # legend
        handles, labels = ax.get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=population_size)
        # save figures to directory
        plt.savefig(fname=Path(save_directory, "multi_plot.png"), format='png', transparent=False)
        plt.savefig(fname=Path(save_directory, "multi_plot.svg"), format='svg', transparent=True)
        # clear current figure and axes
        plt.clf()