import itertools
from functools import partial
from typing import Sequence, Tuple
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.font_manager
import matplotlib.ticker as ticker
from matplotlib import rc
from matplotlib.lines import Line2D

rc('font', family='serif')
rc('font', serif='Computer Modern Roman')
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage[T1]{fontenc} \catcode`\_=12")

from pbt.database import ReadOnlyDatabase
from pbt.utils.iterable import flatten_dict

origin_path = Path('F:/PBT_DATA/2020-06-11')
checkpoints_path = Path(origin_path, 'checkpoints')
data_path = Path(origin_path, 'data')
plots_path = Path(origin_path, 'plots')
box_plots_path = Path(plots_path, 'box')
line_plots_path = Path(plots_path, 'line')
time_plots_path = Path(plots_path, 'time')
table_path = Path(origin_path, 'tables')

DOCUMENT_MAX_COLUMN_WIDTH = 5.92

evolvers = ['pbt', 'de', 'shade', 'lshade']
evolver_tag = 'evolver'

models = ['mlp', 'lenet5']

datasets = ['mnist', 'fashionmnist']
sets = ['train', 'eval', 'test']

eval_metric = 'test_f1'
single_metrics = ['cce', 'f1', 'acc']
metrics = [
    'train_cce', 'eval_cce', 'test_cce',
    'train_f1', 'eval_f1', 'test_f1',
    'train_acc', 'eval_acc', 'test_acc']
metric_rename_dict = {
    'train_cce': 'train: cce',
    'eval_cce': 'valid: cce',
    'test_cce': 'test: cce',
    'train_f1': 'train: f1 score',
    'eval_f1': 'valid: f1 score',
    'test_f1': 'test: f1 score',
    'train_acc': 'train: accuracy',
    'eval_acc': 'valid: accuracy',
    'test_acc': 'test: accuracy'}

time_keys = ['time_training', 'time_testing', 'time_evolving']
time_rename_dict = {
    'time_training': 'training time',
    'time_testing': 'testing time',
    'time_evolving': 'evolving time'}

def ylim_from_df(df: pd.DataFrame, is_increasing: bool, strength: float = 1.5, pad_top: float = 0.1, pad_bottom: float = 0.1) -> Tuple[float, float]:
    if df.empty:
        return
    df = df.dropna(inplace=False, how='any')
    # calculate statistics
    data_max = df.max()
    data_min = df.min()
    data_mean = df.values.mean()
    data_std = df.values.std(ddof=1)
    # identify outliers
    cut_off = data_std * strength
    lower = data_mean - cut_off
    upper = data_mean + cut_off
    if is_increasing:
        return lower, data_max + (data_max - lower) * pad_top
    else:
        return data_min + (data_min - upper) * pad_bottom, upper

def create_dataframes():
    directories = list(checkpoints_path.glob('*'))
    num_tasks = len(directories)
    data_path.mkdir(parents=True, exist_ok=True)
    for task_index, task_path in enumerate(directories, 1):
        task = task_path.name
        df_result = pd.DataFrame()
        print(f"Task {task_index} of {num_tasks}: {task}")
        directories = list(task_path.glob('*'))
        for directory_index, directory in enumerate(directories):
            evolver = directory.name.split("batch64_", 1)[1]
            database_paths = list(directory.glob('*'))
            for database_index, database_path in enumerate(database_paths, 1):
                print(f"-- ({directory_index*len(database_paths)+database_index}/{len(database_paths)*len(directories)}) {database_path}...")
                database = ReadOnlyDatabase(database_path=database_path, read_function=torch.load)
                best = max(database.get_last())
                plot_folder = Path(database_path, 'results', 'plots')
                dataframes = list()
                for csv_file in plot_folder.glob('*.csv'):
                    df_csv = pd.read_csv(csv_file, index_col='steps')
                    series = df_csv[str(best.uid)]
                    name = csv_file.stem.replace('_dots', '').replace('hp_optimizer_', '').replace('loss_', '')
                    dataframes.append(series.rename(name))
                df_member = pd.concat(dataframes, axis=1, sort=False)
                df_member[evolver_tag] = evolver
                df_member['database'] = database_path.name
                df_member.set_index(['database', evolver_tag], append=True, inplace=True)
                df_member.sort_index(inplace=True)
                df_result = df_result.append(df_member)
        df_result.to_csv(Path(data_path, f"{task}.csv"))

def plot_boxplot_figure(df: pd.DataFrame, group_by: str, groups: list, columns: Sequence[str], **kwargs):
    colors = mcolors.TABLEAU_COLORS
    # remove unwanted columns
    df = df[df[group_by].isin(groups)]
    # sort in correct group order
    df[group_by] = pd.Categorical(df[group_by], groups)
    df.sort_values(by=group_by, inplace=True)
    # create figure
    figure, axes = plt.subplots(**kwargs)
    boxplots = df.boxplot(
        ax=axes, by=group_by, column=columns,
        widths=0.5, return_type='both', showfliers=False, vert=True, patch_artist=False)
    for boxplot in boxplots.values:
        boxplot.ax.set_xlabel('')
        boxplot.ax.tick_params(axis='x', labelrotation=0, pad=15, labelsize='small', labelcolor='white')
        # fill with colors
        for lines, color in zip(boxplot.lines['boxes'], itertools.cycle(colors)):
            lines.set_color(color)
    # layout
    figure.tight_layout(h_pad=0.8, w_pad=0.8)
    figure.legend(
        loc='lower center', ncol=len(groups), frameon=False,
        handles=[
            mpatches.Patch(fill=False, edgecolor=color, label=category)
            for category, color in zip(groups, itertools.cycle(colors))])
    plt.margins(0.2)
    plt.suptitle('')
    return figure
    
# BOXPLOT: loss for train, eval and test across steps for pbt, de, shade, lshade on the average
def create_box_plots():
    directories = list(data_path.glob('*'))
    box_plots_path.mkdir(parents=True, exist_ok=True)
    columns = ['train_cce', 'eval_cce', 'test_cce', 'train_f1', 'eval_f1', 'test_f1', 'train_acc', 'eval_acc', 'test_acc']
    for index, task_path in enumerate(directories):
        task = task_path.stem
        evolver_tag = 'evolver'
        df = pd.read_csv(task_path)
        index_of_last_entries = df.groupby(['database'])['steps'].transform(max) == df['steps']
        df = df[index_of_last_entries]
        if df.empty:
            continue
        # create all vs all plot
        for i in range(1, len(evolvers)):
            evolver_set = evolvers[:i + 1]
            file_name = f"{task}_{'_vs_'.join(evolver_set)}_box"
            print(f"-- ({index + 1} of {len(directories)}) creating plot for {file_name}...")
            figure_height = 3
            figure_width = 4.8 + (DOCUMENT_MAX_COLUMN_WIDTH - 4.8) * ((i - 1) / (len(evolvers) - 2))
            figure = plot_boxplot_figure(
                df=df, group_by=evolver_tag, groups=evolver_set, columns=columns,
                figsize=(figure_width, figure_height), nrows=3, ncols=3, sharex=True)
            figure.savefig(fname=Path(box_plots_path, f"{file_name}.png"), format='png', transparent=False)
            figure.savefig(fname=Path(box_plots_path, f"{file_name}.pdf"), format='pdf', transparent=True)
            figure.clf()
            del figure
            plt.close('all')

# PLOT: loss for train, eval and test across steps for pbt, de, shade, lshade on the median best example
def create_line_plots(mode: str = 'mean'):
    colors = mcolors.TABLEAU_COLORS
    figure_height = 5
    figure_width = DOCUMENT_MAX_COLUMN_WIDTH
    line_plots_path.mkdir(parents=True, exist_ok=True)
    directories = list(data_path.glob('*'))
    metrics_dict = {
        'cce': ('train_cce', 'eval_cce', 'test_cce'),
        'f1': ('train_f1', 'eval_f1', 'test_f1'),
        'acc': ('train_acc', 'eval_acc', 'test_acc')}
    for task_index, task_path in enumerate(directories, 1):
        task = task_path.stem
        print(f"-- ({task_index} of {len(directories)}) creating plot for {task}...")
        # read data
        df = pd.read_csv(task_path)
        if df.empty:
            raise Exception(f"data on path {task_path} was empty")
        # remove unwanted columns
        df = df[df[evolver_tag].isin(evolvers)]
        if mode == 'best':
            # keep last entries for each
            index_of_last_entries = df.groupby(['database'])['steps'].transform(max) == df['steps']
            df_last = df[index_of_last_entries]
            # keep best entries for each evolver
            index_of_best_entries = df_last.groupby([evolver_tag])[eval_metric].transform(max) == df_last[eval_metric]
            df_best = df_last[index_of_best_entries]
            # select all entries from best database
            df_plotable = df.loc[df['database'].isin(df_best['database'])]
        elif mode == 'mean':
            df_plotable = df.groupby([evolver_tag, 'steps']).mean().reset_index()
        else:
            raise NotImplementedError()
        for metric_index, (metric_type, metric_group) in enumerate(metrics_dict.items(), 1):
            n_metrics = len(metric_group)
            print(f"---- ({metric_index} of {len(metrics_dict)}) creating plot for {metric_type}...")
            file_name = f"{task}_{metric_type}_line"
            figure = plt.figure(figsize=(figure_width, figure_height))
            gs = figure.add_gridspec(
                nrows=n_metrics + 1, ncols=2,
                width_ratios=[1, 0.5],
                height_ratios=[1]*n_metrics + [0.5])
            for index, metric in enumerate(metric_group):
                left_ax = figure.add_subplot(gs[index, 0])
                right_ax = figure.add_subplot(gs[index, 1])
                # plot
                for evolver in evolvers:
                    df_by_evolver = df_plotable.loc[df_plotable[evolver_tag] == evolver]
                    df_by_evolver.plot(x='steps', y=metric, kind='line', legend=False, ax=left_ax, grid=False)
                    df_by_evolver.plot(x='steps', y=metric, kind='line', legend=False, ax=right_ax, grid=False)
                # draw grid
                left_ax.grid('on', which='major', axis='both')
                right_ax.grid('on', which='major', axis='both')
                # set ax title
                ax_title = figure.add_subplot(gs[index, :])
                ax_title.axis('off')
                left_ax.set_title('')
                right_ax.set_title('')
                ax_title.set_title(metric_rename_dict[metric])
                # remove x axis labels
                left_ax.set_xlabel('')
                right_ax.set_xlabel('')
                if index < len(metric_group) - 1:
                    left_ax.set_xticklabels([])
                    right_ax.set_xticklabels([])
                # set y limits according to metric type
                ylim_boundaries = partial(ylim_from_df, df=df_by_evolver[metric], strength=1.0)
                if metric_type == 'cce':
                    boundaries = ylim_boundaries(is_increasing=False)
                elif metric_type == 'f1':
                    boundaries = ylim_boundaries(is_increasing=True)
                elif metric_type == 'acc':
                    boundaries = ylim_boundaries(is_increasing=True)
                else:
                    raise NotImplementedError(f"metric '{metric}' is not expected.")
                left_ax.set_ylim(boundaries)
                right_ax.set_ylim(boundaries)
                # set x limits
                left_ax.set_xlim(left=0, right=10500)
                right_ax.set_xlim(left=10500, right=23000)
                # set y axis ticks
                left_ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                right_ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                # remove spines
                left_ax.spines['right'].set_visible(False)
                right_ax.spines['left'].set_visible(False)
                # set tick labels
                left_ax.yaxis.tick_left()
                right_ax.yaxis.tick_right()
                ticks = right_ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
            # create legends
            ax_legend = figure.add_subplot(gs[n_metrics, :])
            ax_legend.axis('off')
            ax_legend.legend(
                loc='center', ncol=len(evolvers), frameon=True,
                handles=[Line2D([0], [0], color=color, linewidth=2, label=evolver) for evolver, color in zip(evolvers, itertools.cycle(colors))])
            # set figure layout
            figure.tight_layout(h_pad=0.8, w_pad=0.1)
            plt.margins(0.4)
            # add x axis label
            figure.text(0.5, 0.125, 'steps', ha='center')
            # save figure
            figure.savefig(fname=Path(line_plots_path, f"{file_name}.png"), format='png', transparent=False)
            figure.savefig(fname=Path(line_plots_path, f"{file_name}.pdf"), format='pdf', transparent=True)
            figure.clf()
            del figure
            plt.close('all')

# TABLE: loss for train, eval and test across steps for pbt, de, shade, lshade on the average with std and min and max
def create_table_df():
    table_path.mkdir(parents=True, exist_ok=True)
    directories = list(data_path.glob('*'))
    dataframes = {csv_file.stem: pd.read_csv(csv_file) for csv_file in directories}
    for metric in single_metrics:
        multi_index = pd.MultiIndex.from_product([datasets, models, sets, evolvers], names=['Dataset', 'Model', 'Set', 'Algorithm'])
        result_table_df = pd.DataFrame(columns=['mean', 'std', 'min', 'max'], index=multi_index)
        for task_index, (task, df) in enumerate(dataframes.items(), 1):
            if df.empty:
                raise Exception(f"data on task {task} was empty")
            dataset, model = tuple(task.split('_'))
            # remove unwanted columns
            df = df[df[evolver_tag].isin(evolvers)]
            # keep last entries for each
            index_of_last_entries = df.groupby(['database'])['steps'].transform(max) == df['steps']
            df_last = df[index_of_last_entries]
            # remove unwanted columns
            score_df = df_last[metrics + [evolver_tag]]
            # create statistics
            df_mean = score_df.groupby([evolver_tag]).mean().transpose()
            df_std = score_df.groupby([evolver_tag]).std().transpose()
            df_min = score_df.groupby([evolver_tag]).min().transpose()
            df_max = score_df.groupby([evolver_tag]).max().transpose()
            for stat_type, df_stat in {'mean': df_mean, 'std':df_std, 'min':df_min, 'max':df_max}.items():
                df_stat.rename_axis('metric', inplace=True)
                df_stat.reset_index(inplace=True)
                df_key = pd.DataFrame(df_stat['metric'].str.split('_').tolist(), columns = ['set','metric'])
                df_stat.drop(columns='metric', inplace=True)
                df_stat = pd.concat([df_key, df_stat], axis=1)
                df_stat.set_index(['set','metric'], inplace=True)
                for evolver in evolvers:
                    for s in sets:
                        stat = df_stat.loc[s, metric][evolver]
                        result_table_df.loc[dataset, model, s, evolver][stat_type] = stat
                        continue
        result_table_df.rename(index={
            'lenet5': 'LeNet5',
            'mlp': 'MLP',
            'mnist': 'MNIST',
            'fashionmnist': 'FashionMNIST',
            'pbt': 'PBT',
            'de': 'PBT-DE',
            'shade': 'PBT-SHADE',
            'lshade': 'PBT-LSHADE',
            'eval': 'valid',
            },
            inplace=True)
        file_name= f"{metric}_score_table"
        result_table_df.to_csv(Path(table_path, f"{file_name}.csv"), index=True, float_format='%.5f')
        evolvers_gls = [f"\gls{'{'}{evolver}{'}'}" for evolver in evolvers]
        printable_metrics = {'acc': 'Accuracy', 'f1': 'F1 Score', 'cce': 'CCE'}
        result_table_df.to_latex(
            Path(table_path, f"{file_name}.tex"),
            index=True, multirow=True, multicolumn=True,
            caption=f"The peformance evaluation by {printable_metrics[metric]} of {', '.join(evolvers_gls[:-1])} and {evolvers_gls[-1]}.",
            label=f"tab:{metric}_score_table")

# sum of all member time spent each step for pbt, de, shade and lshade
def create_time_line_plots():
    colors = mcolors.TABLEAU_COLORS
    figure_height = 6
    figure_width = DOCUMENT_MAX_COLUMN_WIDTH
    time_plots_path.mkdir(parents=True, exist_ok=True)
    directories = list(data_path.glob('*'))
    for task_index, task_path in enumerate(directories, 1):
        task = task_path.stem
        print(f"-- ({task_index} of {len(directories)}) creating plot for {task}...")
        # read data
        df = pd.read_csv(task_path)
        if df.empty:
            raise Exception(f"data on path {task_path} was empty")
        # remove unwanted columns
        df = df[df[evolver_tag].isin(evolvers)]
        df_plotable = df.groupby([evolver_tag, 'steps']).mean().reset_index()
        df_plotable['time_sum'] = df_plotable[time_keys].sum(axis=1)
        file_name = f"{task}_time_line"
        figure = plt.figure(figsize=(figure_width, figure_height))
        time_columns = time_keys + ['time_sum']
        gs = figure.add_gridspec(
            nrows=len(time_columns) + 1, ncols=2,
            width_ratios=[1, 0.5],
            height_ratios=[1]*len(time_columns) + [0.5])
        for time_index, time_type in enumerate(time_columns):
            print(f"---- ({time_index + 1} of {len(time_columns)}) creating plot for '{time_type}'...")
            left_ax = figure.add_subplot(gs[time_index, 0])
            right_ax = figure.add_subplot(gs[time_index, 1])
            # plot
            for evolver in evolvers:
                df_by_evolver = df_plotable.loc[df_plotable[evolver_tag] == evolver]
                df_by_evolver.plot(x='steps', y=time_type, kind='line', legend=False, ax=left_ax, grid=False)
                df_by_evolver.plot(x='steps', y=time_type, kind='line', legend=False, ax=right_ax, grid=False)
            # draw grid
            left_ax.grid('on', which='major', axis='both')
            right_ax.grid('on', which='major', axis='both')
            # set ax title
            ax_title = figure.add_subplot(gs[time_index, :])
            ax_title.axis('off')
            left_ax.set_title('')
            right_ax.set_title('')
            if time_type != 'time_sum':
                ax_title.set_title(f"average {time_rename_dict[time_type]} for each member (in seconds)")
            else:
                ax_title.set_title(f"average total generation time for each member (in seconds)")
            # remove x axis labels
            left_ax.set_xlabel('')
            right_ax.set_xlabel('')
            if time_index < len(time_columns) - 1:
                left_ax.set_xticklabels([])
                right_ax.set_xticklabels([])
            # set x limits
            left_ax.set_xlim(left=0, right=10500)
            right_ax.set_xlim(left=10500, right=23000)
            # set y axis ticks
            left_ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            right_ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            # remove spines
            left_ax.spines['right'].set_visible(False)
            right_ax.spines['left'].set_visible(False)
            # set tick labels
            left_ax.yaxis.tick_left()
            right_ax.yaxis.tick_right()
            ticks = right_ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
        # create legends
        ax_legend = figure.add_subplot(gs[len(time_columns), :])
        ax_legend.axis('off')
        ax_legend.legend(
            loc='center', ncol=len(evolvers), frameon=True,
            handles=[Line2D([0], [0], color=color, linewidth=2, label=evolver) for evolver, color in zip(evolvers, itertools.cycle(colors))])
        # set figure layout
        figure.tight_layout(h_pad=0.8, w_pad=0.1)
        plt.margins(0.4)
        # add x axis label
        figure.text(0.5, 0.1, 'steps', ha='center')
        # save figure
        figure.savefig(fname=Path(time_plots_path, f"{file_name}.png"), format='png', transparent=False)
        figure.savefig(fname=Path(time_plots_path, f"{file_name}.pdf"), format='pdf', transparent=True)
        figure.clf()
        del figure
        plt.close('all')
# table of total execution time for pbt, de, shade and lshade

# 10, 20, 30, 50
# loss plot of average results for population sizes with pbt and lshade
# table of average results for population sizes with pbt and lshade

# 50, 100, 250, 500
# loss plot of average results for step sizes with pbt and lshade
# table of average results for step sizes with pbt and lshade


if __name__ == "__main__":
    create_dataframes()
    create_box_plots()
    create_line_plots()
    create_table_df()
    create_time_line_plots()