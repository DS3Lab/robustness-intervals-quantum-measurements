import os
import pandas as pd
from tabulate import tabulate

from lib.compute_bounds import compute_bounds_gramian_eigval, compute_bounds_sdp, compute_bounds_gramian_expectation
from lib.compute_bounds import CLIQUE_PARTITION


def print_latex_table(results_dir):
    stats_df = pd.read_pickle(os.path.join(results_dir, 'all_statistics.pkl'))
    print(tabulate(stats_df, headers='keys', tablefmt='presto', floatfmt=".9f"))
    df1 = compute_bounds_gramian_eigval(stats_df, confidence_level=0.01)
    df2 = compute_bounds_sdp(stats_df, confidence_level=0.01, partition=CLIQUE_PARTITION)
    df3 = compute_bounds_gramian_expectation(stats_df, confidence_level=0.01, partition=CLIQUE_PARTITION)

    df1 = df1.rename(columns={'lower_bound': 'gram_eigval_lower_bound', 'upper_bound': 'gram_eigval_upper_bound',
                              'vqe_energy': 'gram_eigval_energy'})
    df2 = df2.rename(columns={'lower_bound': 'sdp_lower_bound', 'upper_bound': 'sdp_upper_bound',
                              'vqe_energy': 'sdp_energy'})
    df3 = df3.rename(columns={'lower_bound': 'gram_expectation_lower_bound',
                              'upper_bound': 'gram_expectation_upper_bound',
                              'vqe_energy': 'gram_expectation_energy'})

    plot_df = pd.merge(pd.merge(df1, df2, on=['r', 'exact', 'fidelity']), df3, on=['r', 'exact', 'fidelity'])
    print(tabulate(plot_df, headers='keys', tablefmt='presto', floatfmt=".9f"))

    latex_table = pandas_to_latex(plot_df)
    print(latex_table)


def pandas_to_latex(df: pd.DataFrame):
    table = ""
    row_str = "${0:.2f}$ & ${1:.5f}$ & ${2:.5f}$ & ${3:.3f}$ && ${4:.5f}$ & ${5:.5f}$ && ${6:.5f}$ && ${7:.5f}$ & ${8:.5f}$\\\\" + "\n"

    for idx, row in df.iterrows():
        table += row_str.format(
            idx, row['exact'], row['gram_eigval_energy'], row['fidelity'], row['gram_eigval_lower_bound'],
            row['gram_eigval_upper_bound'], row['gram_expectation_lower_bound'], row['sdp_lower_bound'],
            row['sdp_upper_bound'])

    return table
