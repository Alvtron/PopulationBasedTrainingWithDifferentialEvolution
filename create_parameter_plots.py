from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib import rc
from matplotlib.lines import Line2D

# set matplotlib settings
rc('font', family='serif')
rc('font', serif='Computer Modern Roman')
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage[T1]{fontenc} \catcode`\_=12")

from pbt.database import ReadOnlyDatabase
from pbt.utils.iterable import flatten_dict

origin_path = Path('F:/PBT_DATA/2020-06-27')
checkpoints_path = Path(origin_path, 'checkpoints')
data_path = Path(origin_path, 'data')
plots_path = Path(origin_path, 'plots')
bar_plots_path = Path(plots_path, 'bar')

DOCUMENT_MAX_COLUMN_WIDTH = 5.92

procedure_tag = 'procedure'
procedures = ['PBT', 'PBT-SHADE']

population_size_tag = 'N'
population_sizes = [10, 20, 30, 40, 50, 60]

metrics = ['test_cce', 'test_f1', 'test_acc']
metrics_full_name = ['CCE score', 'F1 score', 'Accuracy']
evolvers_rename_dict = {
    'pbt': 'PBT',
    'de': 'PBT-DE',
    'shade': 'PBT-SHADE',
    'lshade': 'PBT-LSHADE'}

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
            procedure = directory.name.split("batch64_", 1)[1]
            population_size = int(directory.name.split("_train", 1)[0].replace('p', ''))
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
                df_member[procedure_tag] = evolvers_rename_dict[procedure]
                df_member[population_size_tag] = population_size
                df_member['database'] = database_path.name
                df_member.set_index(['database', procedure_tag], append=True, inplace=True)
                df_member.sort_index(inplace=True)
                df_result = df_result.append(df_member)
        df_result.to_csv(Path(data_path, f"{task}.csv"))

# loss plot of average results for population sizes with pbt and lshade
# table of average results for population sizes with pbt and lshade
def create_bar_plots():
    directories = list(data_path.glob('*'))
    bar_plots_path.mkdir(parents=True, exist_ok=True)
    figure_height = 4
    figure_width = DOCUMENT_MAX_COLUMN_WIDTH
    ylim_dict = {'test_cce': (0.26, 0.285), 'test_f1': (0.885, 0.895), 'test_acc': (89.5, 90.5)}
    for index, task_path in enumerate(directories):
        task = task_path.stem
        evolver_tag = 'evolver'
        df = pd.read_csv(task_path)
        index_of_last_entries = df.groupby(['database'])['steps'].transform(max) == df['steps']
        df = df[index_of_last_entries]
        if df.empty:
            continue
        df = df.groupby([procedure_tag, population_size_tag, 'steps']).mean().reset_index()
        # create figure
        figure = plt.figure(figsize=(figure_width, figure_height))
        gs = figure.add_gridspec(nrows=4, ncols=1, hspace=0.2, height_ratios=[1]*3 + [0.8])
        for index, (metric, metric_full_name) in enumerate(zip(metrics, metrics_full_name)):
            ax = figure.add_subplot(gs[index])
            data = {procedure: df.loc[df[procedure_tag] == procedure][metric].values for procedure in procedures}
            df_plottable = pd.DataFrame(data, index=population_sizes)
            ax = df_plottable.plot(ax=ax, kind='bar', rot=0, color=['tab:blue', 'tab:green'], legend=False, ylim=ylim_dict[metric])
            ax.set_ylabel(metric_full_name)
            if metric == 'test_acc':
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            else:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            if index < len(metrics) - 1:
                ax.set_xticks([])
        ax.set_xlabel('population size')
        ax_legend = figure.add_subplot(gs[-1])
        ax_legend.axis('off')
        ax_legend.legend(
            loc='lower center', ncol=len(procedures), frameon=False,
            handles=[Line2D([0], [0], color=color, linewidth=2, label=evolver) for evolver, color in zip(procedures, ['tab:blue', 'tab:green'])])
        plt.subplots_adjust(top=0.95, bottom=0.00, left=0.12, right=0.95)
        figure.savefig(fname=Path(bar_plots_path, f"population_size_bar.png"), format='png', transparent=False)
        figure.savefig(fname=Path(bar_plots_path, f"population_size_bar.pdf"), format='pdf', transparent=True)
        figure.clf()
        plt.close('all')

if __name__ == "__main__":
    #create_dataframes()
    create_bar_plots()