import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from pathlib import Path
from database import ReadOnlyDatabase
from evaluator import Evaluator
from hyperparameters import Hyperparameter, Hyperparameters
from utils import clip

class Analyzer(object):
    def __init__(self, database : ReadOnlyDatabase, evaluator : Evaluator):
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

    def create_statistics(self, save_directory):
        population_entries = self.database.to_dict()
        # get member statistics
        checkpoint_summaries = dict()
        for entry_id, entries in population_entries.items():
            num_entries = len(entries.values())
            total_eval_score = sum(c.eval_score for c in entries.values())
            total_train_time = sum(c.train_time for c in entries.values())
            total_eval_time = sum(c.eval_time for c in entries.values())
            total_evolve_time = sum(c.evolve_time for c in entries.values() if c.evolve_time)
            checkpoint_summaries[entry_id] = {
                'num_entries':num_entries,
                'steps':max(c.steps for c in entries.values()),
                'epochs':max(c.epochs for c in entries.values()),
                'max_eval_score':max(c.eval_score for c in entries.values()),
                'avg_eval_score':total_eval_score/num_entries,
                'total_eval_score':total_eval_score,
                'max_train_time':max(c.train_time for c in entries.values()),
                'avg_train_time':total_train_time/num_entries,
                'total_train_time':total_train_time,
                'max_eval_time':max(c.eval_time for c in entries.values()),
                'avg_eval_time':total_eval_time/num_entries,
                'total_eval_time':total_eval_time,
                'max_evolve_time':max(c.evolve_time for c in entries.values() if c.evolve_time),
                'avg_evolve_time':total_evolve_time/num_entries,
                'total_evolve_time':total_evolve_time
            }
        # print member statistics
        for entry_id, checkpoint_summary in checkpoint_summaries.items():
            print(f"Statistics for member {entry_id}:")
            for tag, statistic in checkpoint_summary.items():
                print(f"{tag}:{statistic}")
        # get global statistics
        print(f"Statistics for member {entry_id}:")
        global_avg_num_entries=sum(d['num_entries'] for d in checkpoint_summaries.values() if d)/len(checkpoint_summaries)
        global_total_eval_score=sum(d['total_eval_score'] for d in checkpoint_summaries.values() if d)
        global_total_train_time=sum(d['total_train_time'] for d in checkpoint_summaries.values() if d)
        global_total_eval_time=sum(d['total_eval_time'] for d in checkpoint_summaries.values() if d)
        global_total_evolve_time=sum(d['total_evolve_time'] for d in checkpoint_summaries.values() if d)
        global_checkpoint_summaries = {
            'avg_num_entries':global_avg_num_entries,
            'total_eval_score':global_total_eval_score,
            'total_train_time':global_total_train_time,
            'total_eval_time':global_total_eval_time,
            'total_evolve_time':global_total_evolve_time,
            'max_eval_score':max(d['max_eval_score'] for d in checkpoint_summaries.values() if d),
            'max_train_time':max(d['max_train_time'] for d in checkpoint_summaries.values() if d),
            'max_eval_time':max(d['max_eval_time'] for d in checkpoint_summaries.values() if d),
            'max_evolve_time':max(d['max_evolve_time'] for d in checkpoint_summaries.values() if d),
            'avg_eval_score':global_total_eval_score/global_avg_num_entries,
            'avg_train_time':global_total_train_time/global_avg_num_entries,
            'avg_eval_time':global_total_eval_time/global_avg_num_entries,
            'avg_evolve_time':global_total_evolve_time/global_avg_num_entries
        }
        # print global statistics
        for tag, statistic in global_checkpoint_summaries.items():
            print(f"{tag}:{statistic}")

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
            for entry in entries.values(): # for each entry
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