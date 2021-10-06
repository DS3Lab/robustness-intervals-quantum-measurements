import pandas as pd
import numpy as np
from scipy import stats

from lib.robustness_interval import SDPInterval, GramianExpectationBound, GramianEigenvalueInterval

CLIQUE_PARTITION = 'clique-partition'
PAULI_PARTITION = 'pauli-partition'
NO_PARTITION = 'no-partition'  # i.e. H is not partitioned into terms


def compute_bounds_sdp(stats_df: pd.DataFrame, partition: str, confidence_level: float):
    if partition == CLIQUE_PARTITION:
        df_keys1 = ['paulicliques', 'paulicliques_coeffs', 'paulicliques_eigenvalues']
        df_keys2 = ['paulicliques_expectations', 'paulicliques_variances']
    elif partition == PAULI_PARTITION:
        df_keys1 = ['pauli_strings', 'pauli_coeffs', 'pauli_eigenvalues']
        df_keys2 = ['pauli_expectations', 'pauli_variances']
    else:
        raise ValueError(f'unknown partition {partition} for sdp bounds')

    precomp_stats_keys1 = ['pauligroups', 'pauligroups_coeffs', 'pauligroups_eigenvalues']
    precomp_stats_keys2 = ['pauligroups_expectations', 'pauligroups_variances']

    data = compute_bounds(stats_df, precomp_stats_keys1, precomp_stats_keys2, df_keys1, df_keys2, SDPInterval,
                          confidence_level)
    df = pd.DataFrame(data, columns=['r', 'exact', 'fidelity', 'vqe_energy', 'lower_bound', 'upper_bound'])
    df = df.sort_values('r')
    df = df.set_index('r')
    return df


def compute_bounds_gramian_expectation(stats_df: pd.DataFrame, partition: str, confidence_level: float):
    if partition == CLIQUE_PARTITION:
        df_keys1 = ['paulicliques', 'paulicliques_coeffs', 'paulicliques_eigenvalues']
        df_keys2 = ['paulicliques_expectations', 'paulicliques_variances']
    elif partition == PAULI_PARTITION:
        df_keys1 = ['pauli_strings', 'pauli_coeffs', 'pauli_eigenvalues']
        df_keys2 = ['pauli_expectations', 'pauli_variances']
    else:
        raise ValueError(f'unknown partition {partition} for sdp bounds')

    precomp_stats_keys1 = ['pauligroups', 'pauligroups_coeffs', 'pauligroups_eigenvalues']
    precomp_stats_keys2 = ['pauligroups_expectations', 'pauligroups_variances']

    data = compute_bounds(stats_df, precomp_stats_keys1, precomp_stats_keys2, df_keys1, df_keys2,
                          GramianExpectationBound, confidence_level)
    df = pd.DataFrame(data, columns=['r', 'exact', 'fidelity', 'vqe_energy', 'lower_bound', 'upper_bound'])
    df = df.sort_values('r')
    df = df.set_index('r')
    return df


def compute_bounds_gramian_eigval(stats_df: pd.DataFrame, confidence_level: float):
    precomp_stats_keys = df_keys = ['hamiltonian_expectations', 'hamiltonian_variances']
    data = compute_bounds(stats_df, {}, precomp_stats_keys, {}, df_keys, GramianEigenvalueInterval, confidence_level)
    df = pd.DataFrame(data, columns=['r', 'exact', 'fidelity', 'vqe_energy', 'lower_bound', 'upper_bound'])
    df = df.sort_values('r')
    df = df.set_index('r')
    return df


def compute_bounds(stats_df, precomputed_stats_keys1, precomputed_stats_keys2, df_keys1, df_keys2, interval_class,
                   confidence_level):
    data = []

    for r in stats_df.index:
        true_fidelity_vals = stats_df.loc[r, 'gs_fidelity_reps']
        nreps = stats_df.loc[r, 'nreps']

        intervals = []
        precomputed_stats = {k1: stats_df.loc[r][k2] for k1, k2 in zip(precomputed_stats_keys1, df_keys1)}

        for i in range(nreps):
            precomputed_stats = {**precomputed_stats,
                                 **{k1: stats_df.loc[r][k2][i] for k1, k2 in zip(precomputed_stats_keys2, df_keys2)}}
            intervals.append(interval_class(fidelity=true_fidelity_vals[i], precomputed_stats=precomputed_stats))

        if nreps == 1:
            lower_bound = intervals[0].lower_bound
            upper_bound = intervals[0].upper_bound
            energy = intervals[0].expectation
        else:
            all_lower_bounds = [i.lower_bound for i in intervals]
            all_upper_bounds = [i.upper_bound for i in intervals]
            energy = np.mean([i.expectation for i in intervals])

            # compute one-sided confidence intervals
            lower_bound_mean = np.mean(all_lower_bounds)
            lower_bound_var = np.var(all_lower_bounds, ddof=1, dtype=np.float64)
            lower_bound = lower_bound_mean - np.sqrt(lower_bound_var / nreps) * stats.t.ppf(
                q=1 - confidence_level, df=nreps - 1)

            # compute one-sided confidence intervals
            upper_bound_mean = np.mean(all_upper_bounds)
            upper_bound_var = np.var(all_upper_bounds, ddof=1, dtype=np.float64)
            upper_bound = upper_bound_mean + np.sqrt(upper_bound_var / nreps) * stats.t.ppf(
                q=1 - confidence_level, df=nreps - 1)

        data.append([r, stats_df.loc[r, 'E0'], np.mean(true_fidelity_vals), energy, lower_bound, upper_bound])

    return data
