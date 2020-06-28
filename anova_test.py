import itertools
from functools import partial
from typing import Sequence, Tuple
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
import researchpy as rp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.font_manager
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib import rc
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

rc('font', family='serif')
rc('font', serif='Computer Modern Roman')
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage[T1]{fontenc} \catcode`\_=12")

from pbt.database import ReadOnlyDatabase
from pbt.utils.iterable import flatten_dict

origin_path = Path('F:/PBT_DATA/2020-06-11')
checkpoints_path = Path(origin_path, 'checkpoints')
data_path = Path(origin_path, 'data')
statistic_path = Path(origin_path, 'statistic')

DOCUMENT_MAX_COLUMN_WIDTH = 5.92

evolvers = ['PBT', 'PBT-DE', 'PBT-SHADE', 'PBT-LSHADE']
evolver_tag = 'evolver'
models = ['mlp', 'lenet5']
models_rename_dict = {'mlp': 'MLP', 'lenet5': 'LeNet-5'}
datasets = ['mnist', 'fashionmnist']

def power_of_notation(value: float):
    scientific = f"{value:0.5E}"
    number, exponent = tuple(scientific.split('E'))
    exponent = int(exponent)
    if (exponent < -2):
        return f"${number} \\times 10^" + '{' + str(exponent) + "}$"
    else:
        return f"{value:0.5f}"

# BOXPLOT: loss for train, eval and test across steps for pbt, de, shade, lshade on the average
def perform_full_anova_test():
    directories = list(data_path.glob('*'))
    statistic_path.mkdir(parents=True, exist_ok=True)
    figure_height = 3
    groups = evolvers
    dependent_variable='test_f1'
    statistic_labels = ['statistic', 'p-value']
    multi_index_normality = pd.MultiIndex.from_product([datasets, models, evolvers], names=['Dataset', 'Model', 'Procedure'])
    df_normality_full = pd.DataFrame(columns=statistic_labels, index=multi_index_normality)
    multi_index_dataset_model = pd.MultiIndex.from_product([datasets, models], names=['Dataset', 'Model'])
    df_levene = pd.DataFrame(columns=statistic_labels, index=multi_index_dataset_model)
    df_oneway_anova = pd.DataFrame(columns=statistic_labels, index=multi_index_dataset_model)
    post_hoc_groups = [
        'PBT \& PBT-DE',
        'PBT \& PBT-SHADE',
        'PBT \& PBT-LSHADE',
        'PBT-DE \& PBT-SHADE',
        'PBT-DE \& PBT-LSHADE',
        'PBT-LSHADE \& PBT-SHADE']
    multi_index_post_hoc = pd.MultiIndex.from_product([datasets, models, post_hoc_groups], names=['Dataset', 'Model', 'Procedures'])
    df_post_hoc = pd.DataFrame(columns=['meandiff', 'lower', 'upper', 'reject'], index=multi_index_post_hoc)
    normality_figure, normality_axes = plt.subplots(nrows=4, ncols=4, figsize=(DOCUMENT_MAX_COLUMN_WIDTH, DOCUMENT_MAX_COLUMN_WIDTH), sharex=True)
    for index, task_path in enumerate(directories):
        dataset, model = tuple(task_path.stem.split('_'))
        print(f"-- ({index + 1} of {len(directories)}) creating plot for {task_path}...")
        df = pd.read_csv(task_path)
        index_of_last_entries = df.groupby(['database'])['steps'].transform(max) == df['steps']
        df = df[index_of_last_entries]
        if df.empty:
            continue
        df[evolver_tag] = pd.Categorical(df[evolver_tag], groups)
        df.sort_values(by=evolver_tag, inplace=True)
        df = df.pivot_table(values=dependent_variable, index=df.index, columns=evolver_tag, aggfunc='first')
        df = df.apply(lambda x: pd.Series(x.dropna().values))
        # NORMALITY test
        normality_data = {evolver: stats.shapiro(df[evolver]) for evolver in evolvers}
        df_normality = pd.DataFrame(data=normality_data, index=statistic_labels)
        for column, (ax, evolver) in enumerate(zip(normality_axes[index], evolvers)):
            df_normality_full.loc[dataset, model, evolver] = df_normality[evolver].T
            stats.probplot(df[evolver], dist="norm", plot=ax)
            # set marker size
            ax.get_lines()[0].set_markersize(3.0)
            # set titles
            if index == 0:
                ax.set_title(evolver)
            else:
                ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            # set labels
            if column == 0:
                if dataset == 'mnist':
                    y_label = f"MNIST \n/w {models_rename_dict[model]}"
                elif dataset == 'fashionmnist':
                    y_label = f"FashionMNIST \n /w {models_rename_dict[model]}"
                ax.set_ylabel(y_label)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            else:
                ax.set_yticks([])
        # HOMOGENEITY OF VARIANCE test
        levene_data = stats.levene(*(df[evolver] for evolver in evolvers))
        df_levene.loc[dataset, model] = levene_data
        # ONE-WAY ANOVA
        oneway_result = stats.f_oneway(*(df[evolver] for evolver in evolvers))
        df_oneway_anova.loc[dataset, model] = oneway_result
        # POST-HOC
        stacked_data = df.stack().reset_index()
        stacked_data.rename(columns={'level_0': 'id','evolver': 'procedure', 0:'result'}, inplace=True)
        multi_comparison = MultiComparison(stacked_data['result'], stacked_data['procedure'])
        post_hoc_results = multi_comparison.tukeyhsd(alpha=0.01)
        post_hoc_df = pd.DataFrame(data=post_hoc_results._results_table.data[1:], columns=post_hoc_results._results_table.data[0])
        for index, row in post_hoc_df.iterrows():
            group = f"{row[0]} \& {row[1]}"
            df_post_hoc.loc[dataset, model, group] = row[2:]
    normality_figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    normality_figure.text(0.5, 0.035, "Theoretical quantiles", ha='center')
    normality_figure.savefig(fname=Path(statistic_path, f"normality.png"), format='png', transparent=False)
    normality_figure.savefig(fname=Path(statistic_path, f"normality.pdf"), format='pdf', transparent=True)
    rename_dict = {
        'mnist': 'MNIST',
        'fashionmnist': r'\shortstack[l]{\textit{Fashion}\\MNIST}',
        'lenet5': 'LeNet-5',
        'mlp': 'MLP',
        'pbt': 'PBT',
        'PBT-LSHADE \& PBT-SHADE': 'PBT-SHADE \& PBT-LSHADE'}
    df_normality_full.rename(index=rename_dict, inplace=True)
    df_normality_full.to_latex(
        Path(statistic_path, "normality.tex"),
        index=True, multirow=True, multicolumn=True, escape=False,
        caption=f"The W-test statistic and p-value for each task using the Shapiro-Wilk method to assess whether the data is normally distributed (the null hypothesis). If the p-value is less than the chosen alpha level, the null hypothesis is rejected.",
        label=f"tab:normality_test")
    df_levene.rename(index=rename_dict, inplace=True)
    df_levene.to_latex(
        Path(statistic_path, "levene.tex"),
        index=True, multirow=True, multicolumn=True, escape=False,
        caption=f"The Levene statistic and p-value for each task using the Levene's test method to assess whether the data is equal in variance (the null hypothesis). If the p-value is less than the chosen alpha level, the null hypothesis is rejected.",
        label=f"tab:levene_test")
    df_oneway_anova.rename(index=rename_dict, inplace=True)
    df_oneway_anova.to_latex(
        Path(statistic_path, "oneway_anova.tex"), float_format=power_of_notation,
        index=True, multirow=True, multicolumn=True, escape=False,
        caption=f"The one-way ANOVA test results for each task for assessing whether there is a statistically significant difference between the procedures.",
        label=f"tab:oneway_anova_test")
    df_post_hoc.rename(index=rename_dict, inplace=True)
    df_post_hoc.to_latex(
        Path(statistic_path, "post_hoc.tex"),
        index=True, multirow=True, multicolumn=True, escape=False,
        caption=f"The results from using the Tukey Honestly Significant Difference (HSD) test with a significance level of 0.05, rejecting or accepting whether there is a here is a statistically significant difference between each procedure.",
        label=f"tab:post_hoc_test")
if __name__ == "__main__":
    perform_full_anova_test()