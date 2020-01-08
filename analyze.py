import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pickle
from pathlib import Path
from database import ReadOnlyDatabase
from evaluator import Evaluator
from hyperparameters import Hyperparameter, Hyperparameters
from utils import clip, mergeDict

class Analyzer(object):
    def __init__(self, database : ReadOnlyDatabase):
        self.database = database

    def create_progression_dict(self):
        population_entries = self.database.to_dict()
        checkpoint_progression = dict()
        for entry_id, entries in population_entries.items():
            checkpoint_progression[entry_id] = dict()
            for entry in entries.values():
                entry_dict = entry.__dict__
                for attribute, value in entry_dict.items():
                    if not attribute in checkpoint_progression[entry_id]:
                        checkpoint_progression[entry_id][attribute] = list()
                    checkpoint_progression[entry_id][attribute] += [value]
        return checkpoint_progression

    def test(self, evaluator : Evaluator, limit = None):
        entries = self.database.to_list()
        if limit:
            entries.sort(key=lambda e: e.eval_score, reverse=True)
            entries = entries[:limit]
        for entry in entries:
            entry.test_score = evaluator.eval(entry.model_state)
        return entries

    def create_statistics(self, save_directory, verbose=False):
        population_entries = self.database.to_dict()
        # get member statistics
        checkpoint_summaries = dict()
        for entry_id, entries in population_entries.items():
            num_entries = len(entries.values())
            total_train_loss = sum(c.train_loss for c in entries.values())
            total_eval_score = sum(c.eval_score for c in entries.values())
            total_train_time = sum(c.train_time for c in entries.values())
            total_eval_time = sum(c.eval_time for c in entries.values())
            total_evolve_time = sum(c.evolve_time for c in entries.values() if c.evolve_time)
            checkpoint_summaries[entry_id] = {
                'num_entries':num_entries,
                'steps':max(c.steps for c in entries.values()),
                'epochs':max(c.epochs for c in entries.values()),
                'total_train_time':total_train_time,
                'total_eval_time':total_eval_time,
                'total_evolve_time':total_evolve_time,
                'max_train_loss':max(c.train_loss for c in entries.values()),
                'max_eval_score':max(c.eval_score for c in entries.values()),
                'max_train_time':max(c.train_time for c in entries.values()),
                'max_eval_time':max(c.eval_time for c in entries.values()),
                'max_evolve_time':max(c.evolve_time for c in entries.values() if c.evolve_time),
                'min_train_loss':min(c.train_loss for c in entries.values()),
                'min_eval_score':min(c.eval_score for c in entries.values()),
                'min_train_time':min(c.train_time for c in entries.values()),
                'min_eval_time':min(c.eval_time for c in entries.values()),
                'min_evolve_time':min(c.evolve_time for c in entries.values() if c.evolve_time),
                'avg_train_loss':total_train_loss/num_entries,
                'avg_eval_score':total_eval_score/num_entries,
                'avg_train_time':total_train_time/num_entries,
                'avg_eval_time':total_eval_time/num_entries,
                'avg_evolve_time':total_evolve_time/num_entries
            }
        # save/print member statistics
        for entry_id, checkpoint_summary in checkpoint_summaries.items():
            if verbose: print(f"Statistics for member {entry_id}:")
            with open(f"{save_directory}/{entry_id}_statistics.txt", "a+") as file:
                for tag, statistic in checkpoint_summary.items():
                    info = f"{tag}:{statistic}"
                    if verbose: print(info)
                    file.write(info + "\n")
        # get global statistics
        global_avg_num_entries=sum(d['num_entries'] for d in checkpoint_summaries.values() if d)/len(checkpoint_summaries)
        global_total_train_loss=sum(d['avg_train_loss'] for d in checkpoint_summaries.values() if d)
        global_total_eval_score=sum(d['avg_eval_score'] for d in checkpoint_summaries.values() if d)
        global_total_train_time=sum(d['total_train_time'] for d in checkpoint_summaries.values() if d)
        global_total_eval_time=sum(d['total_eval_time'] for d in checkpoint_summaries.values() if d)
        global_total_evolve_time=sum(d['total_evolve_time'] for d in checkpoint_summaries.values() if d)
        global_checkpoint_summaries = {
            'avg_num_entries':global_avg_num_entries,
            'total_train_time':global_total_train_time,
            'total_eval_time':global_total_eval_time,
            'total_evolve_time':global_total_evolve_time,
            'max_train_loss':max(d['max_train_loss'] for d in checkpoint_summaries.values() if d),
            'max_eval_score':max(d['max_eval_score'] for d in checkpoint_summaries.values() if d),
            'max_train_time':max(d['max_train_time'] for d in checkpoint_summaries.values() if d),
            'max_eval_time':max(d['max_eval_time'] for d in checkpoint_summaries.values() if d),
            'max_evolve_time':max(d['max_evolve_time'] for d in checkpoint_summaries.values() if d),
            'min_train_loss':min(d['min_train_loss'] for d in checkpoint_summaries.values() if d),
            'min_eval_score':min(d['min_eval_score'] for d in checkpoint_summaries.values() if d),
            'min_train_time':min(d['min_train_time'] for d in checkpoint_summaries.values() if d),
            'min_eval_time':min(d['min_eval_time'] for d in checkpoint_summaries.values() if d),
            'min_evolve_time':min(d['min_evolve_time'] for d in checkpoint_summaries.values() if d),
            'avg_train_loss':global_total_train_loss/global_avg_num_entries,
            'avg_eval_score':global_total_eval_score/global_avg_num_entries,
            'avg_train_time':global_total_train_time/global_avg_num_entries,
            'avg_eval_time':global_total_eval_time/global_avg_num_entries,
            'avg_evolve_time':global_total_evolve_time/global_avg_num_entries
        }
        # save/print global statistics
        with open(f"{save_directory}/global_summary.txt", "a+") as file:
            for tag, statistic in global_checkpoint_summaries.items():
                info = f"{tag}:{statistic}"
                if verbose: print(info)
                file.write(info + "\n")

    def create_plot_files(self, save_directory):
        plot_attributes = {'train_loss', 'eval_score', 'test_score', 'train_time', 'eval_time', 'evolve_time'}
        plt.xlabel("steps")
        progression_dict = self.create_progression_dict()
        for attribute in plot_attributes:
            plt.ylabel(attribute)
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
                parameter_values = [hp[param_name].normalized() for hp in progression_dict[id]['hyper_parameters']]
                scores = [score for score in progression_dict[id]['eval_score']]
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
                parameter_values = [hp[param_name].normalized() for hp in progression_dict[id]['hyper_parameters']]
                scores = [score for score in progression_dict[id]['eval_score']]
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