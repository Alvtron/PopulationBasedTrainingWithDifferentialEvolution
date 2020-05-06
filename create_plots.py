import itertools
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.font_manager
from matplotlib import rc
rc('font', family='serif')
rc('font', serif='Computer Modern Roman')
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage[T1]{fontenc} \catcode`\_=12")

from pbt.database import ReadOnlyDatabase
from pbt.utils.iterable import flatten_dict

origin_path = Path('F:\PBT_DATA\eval_random_subset')
checkpoints_path = Path(origin_path, 'checkpoints')
data_path = Path(origin_path, 'data')
plots_path = Path(origin_path, 'plots')

def get_best_member(database):
    best = None
    for member in database:
        if best is None:
            best = member
            continue
        if member.steps < best.steps:
            continue
        if member.steps == best.steps and member < best:
            continue
        best = member
    return best

def create_dataframes():
    directories = list(checkpoints_path.glob('*'))
    num_tasks = len(directories)
    for task_index, task_path in enumerate(directories):
        task = task_path.name
        directory_paths = list(task_path.glob('*'))
        data = pd.DataFrame()
        num_directories = len(directory_paths)
        print(f"Task {task_index} of {num_tasks}")
        for directory_index, directory_path in enumerate(directory_paths):
            print(f"-- ({directory_index}/{num_directories}) {directory_path}...")
            evolver = directory_path.name.split("nfe1200_", 1)[1]
            for database_path in directory_path.glob('*'):
                print(f"---- Importing {database_path.name}...")
                database = ReadOnlyDatabase(database_path=database_path, read_function=torch.load)
                best = get_best_member(database)
                flatten = flatten_dict(best.loss)
                flatten['evolver'] = evolver
                loss_df = pd.DataFrame(flatten, index=[database_path.name])
                data = data.append(loss_df)
        data.to_csv(Path(data_path, f"{task}.csv"))

def plot_boxplot_figure(df : pd.DataFrame, group : str, categories : list, **kwargs):
    colors = mcolors.TABLEAU_COLORS
    df = df.copy(deep=True)
    df = df[df[group].isin(categories)] # remove unwanted columns
    df.evolver = df.evolver.astype("category")
    df.evolver.cat.set_categories(categories, inplace=True)
    df = df.sort_values(by=group)
    df_grp = df.groupby(group).tail(5)
    figure, axes = plt.subplots(**kwargs)
    boxplots = df_grp.boxplot(ax=axes, by=group, column=['train_acc', 'eval_acc', 'test_acc', 'train_cce', 'eval_cce', 'test_cce'], widths=0.5,
        return_type='both', showfliers=False, vert=True, patch_artist=False)
    for boxplot in boxplots.values:
        boxplot.ax.set_xlabel('')
        boxplot.ax.tick_params(axis='x', labelrotation=0, pad=15, labelsize='small', labelcolor='white')
        # fill with colors
        for lines, color in zip(boxplot.lines['boxes'], itertools.cycle(colors)):
            lines.set_color(color)
    # layout
    figure.tight_layout(h_pad=0.8, w_pad=0.8)
    figure.legend(
        loc='lower center', ncol=len(categories), frameon=False,
        handles=[mpatches.Patch(fill=False, edgecolor=color, label=category) for category, color in zip(categories, itertools.cycle(colors))])
    plt.margins(0.2)
    plt.suptitle('')
    return figure

def create_stats():
    directories = list(data_path.glob('*'))
    for index, task_path in enumerate(directories):
        task = task_path.stem
        evolver_tag = 'evolver'
        df = pd.read_csv(task_path, index_col=0)
        if df.empty:
            continue
        # shorten column names
        names = {
            'lshade_conservative': 'lshade_c',
            'lshade_very_conservative': 'lshade_vc'}
        df[evolver_tag].replace(names, inplace=True)
        # define evolvers to plot
        evolvers = ['pbt', 'de', 'shade', 'lshade', 'lshade_c']
        # create all vs all plot
        for i in range(1, len(evolvers)):
            evolver_set = evolvers[:i + 1]
            file_name = f"{task}_{'_vs_'.join(evolver_set)}"
            print(f"-- ({index + 1} of {len(directories)}) creating plot for {file_name}...")
            figure_height = 3
            figure_width = 4.8 + (5.92 - 4.8) * ((i - 1) / (len(evolvers) - 2))
            figure = plot_boxplot_figure(df=df, group=evolver_tag, categories=evolver_set, figsize=(figure_width, figure_height), nrows=2, ncols=3, sharex=True)
            figure.savefig(fname=Path(plots_path, f"{file_name}.png"), format='png', transparent=False)
            figure.savefig(fname=Path(plots_path, f"{file_name}.pdf"), format='pdf', transparent=True)
            figure.clf()
            del figure
            plt.close('all')

# MAIN
#create_dataframes()
create_stats()