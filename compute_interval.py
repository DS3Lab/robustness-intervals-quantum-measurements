import argparse
import os
import pandas as pd
from tabulate import tabulate

from lib.compute_bounds import compute_bounds_gramian_eigval, compute_bounds_sdp, compute_bounds_gramian_expectation
from lib.compute_bounds import CLIQUE_PARTITION, PAULI_PARTITION, NO_PARTITION

GRAMIAN_EIGVAL_METHOD = 'gramian-eigval'
GRAMIAN_EXPECTATION_METHOD = 'gramian-expectation'
SDP_METHOD = 'sdp'

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, required=False, default=res, help='dir from which to load results')
parser.add_argument("--method", type=str, required=False, default=SDP_METHOD,
                    choices=['all', GRAMIAN_EIGVAL_METHOD, GRAMIAN_EXPECTATION_METHOD, SDP_METHOD])
parser.add_argument("--partition", type=str, required=False, default=CLIQUE_PARTITION,
                    choices=[CLIQUE_PARTITION, PAULI_PARTITION, NO_PARTITION])
parser.add_argument("--confidence", type=float, default=1e-2)
args = parser.parse_args()


def main():
    if args.method == GRAMIAN_EIGVAL_METHOD:
        args.partition = NO_PARTITION

    if args.method == 'all':
        stats_df = get_stats_df(args.results_dir, 'all')
    else:
        stats_df = get_stats_df(args.results_dir, args.partition)

    df1 = df2 = df3 = None

    if args.method == GRAMIAN_EIGVAL_METHOD or args.method == 'all':
        df1 = compute_bounds_gramian_eigval(stats_df, args.confidence)
        if args.method != 'all':
            print(tabulate(df1, headers='keys', tablefmt='presto', floatfmt=".9f"))

    if args.method == SDP_METHOD or args.method == 'all':
        df2 = compute_bounds_sdp(stats_df, args.partition, args.confidence)
        if args.method != 'all':
            print(tabulate(df2, headers='keys', tablefmt='presto', floatfmt=".9f"))

    if args.method == GRAMIAN_EXPECTATION_METHOD or args.method == 'all':
        df3 = compute_bounds_gramian_expectation(stats_df, args.partition, args.confidence)
        if args.method != 'all':
            print(tabulate(df3, headers='keys', tablefmt='presto', floatfmt=".9f"))

    if args.method == 'all':
        df1 = df1.rename(columns={'lower_bound': 'gram_eigval_lower_bound', 'upper_bound': 'gram_eigval_upper_bound',
                                  'vqe_energy': 'gram_eigval_energy'})
        df2 = df2.rename(columns={'lower_bound': 'sdp_lower_bound', 'upper_bound': 'sdp_upper_bound',
                                  'vqe_energy': 'sdp_energy'})
        df3 = df3.rename(columns={'lower_bound': 'gram_expectation_lower_bound',
                                  'upper_bound': 'gram_expectation_upper_bound',
                                  'vqe_energy': 'gram_expectation_energy'})

        df = pd.merge(pd.merge(df1, df2, on=['r', 'exact', 'fidelity']), df3, on=['r', 'exact', 'fidelity'])
        print(tabulate(df, headers='keys', tablefmt='presto', floatfmt=".9f"))


def get_stats_df(results_dir: str, partition: str) -> pd.DataFrame:
    stats_file = os.path.join(results_dir, 'all_statistics.pkl')
    if os.path.isfile(stats_file):
        df = pd.read_pickle(stats_file)
        return df

    if partition == 'all':
        raise FileNotFoundError(f"file {stats_file} with statistics not found!")

    if partition == CLIQUE_PARTITION:
        stats_file = os.path.join(results_dir, 'paulicliques_statistics.pkl')

        if not os.path.isfile(stats_file):
            raise FileNotFoundError(f"file {stats_file} with pauliclique statistics not found!")

        df = pd.read_pickle(stats_file)
        return df

    if partition == PAULI_PARTITION:
        stats_file = os.path.join(results_dir, 'pauli_statistics.pkl')

        if not os.path.isfile(stats_file):
            raise FileNotFoundError(f"file {stats_file} with pauli statistics not found!")

        df = pd.read_pickle(stats_file)
        return df

    if partition == NO_PARTITION:
        stats_file = os.path.join(results_dir, 'hmailtonian_statistics.pkl')

        if not os.path.isfile(stats_file):
            raise FileNotFoundError(f"file {stats_file} with hamiltonian statistics not found!")

        df = pd.read_pickle(stats_file)
        return df


if __name__ == '__main__':
    main()
