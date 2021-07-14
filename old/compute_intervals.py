import argparse
import _pickle as pickle
from jax.config import config as jax_config
import multiprocessing as mp
import os
import pandas as pd
import sys
from tabulate import tabulate
import time

import tequila as tq  # noqa

from constants import DATA_DIR
from lib.vqe import make_ansatz
from old.robustness_interval import EigenvalueInterval, ExpectationInterval
from lib.helpers import get_molecule_initializer
from lib.helpers import print_summary, timestamp_human, Logger
from lib.noise_models import get_noise_model

parser = argparse.ArgumentParser()
parser.add_argument("--molecule", type=str, required=True, choices=['h2', 'lih', 'beh2'])
parser.add_argument("--results_dir", type=str, required=True, help='dir from which to load results')
parser.add_argument("--samples", "-s", type=int, default=None)
parser.add_argument("--reps", type=int, default=30)
parser.add_argument("--use_grouping", type=int, default=1, choices=[0, 1])
args = parser.parse_args()

N_PNO = None

geom_strings = {
    'h2': 'H .0 .0 .0\nH .0 .0 {r}',
    'lih': 'H .0 .0 .0\nLi .0 .0 {r}',
    'beh2': 'Be .0 .0 .0\nH .0 .0 {r}\nH .0 .0 -{r}'
}

active_orbitals = {
    'h2': None,
    'lih': None,
    'beh2': None
}

columns = ["r", "exact", "vqe", "gs_fidelity", "lower_mean", "upper_mean", "lower_ci", "upper_ci",
           "vqe_mean", "variance_mean"]

print_table_columns = ["r", "exact", "vqe", "gs_fidelity", "lower_mean", "lower_ci", "upper_mean", "upper_ci",
                       "vqe_mean", "variance_mean"]


def listener(q, df_save_path):
    df_first_moment_interval = pd.DataFrame(columns=columns)
    df_second_moment_interval = pd.DataFrame(columns=columns)
    df_eigen_interval = pd.DataFrame(columns=columns)

    while True:
        data = q.get()

        if data == "kill":
            df_first_moment_interval.sort_values('r', inplace=True)
            df_first_moment_interval.set_index('r', inplace=True)
            df_first_moment_interval.to_pickle(path=os.path.join(df_save_path, 'interval_first_moment.pkl'))

            df_second_moment_interval.sort_values('r', inplace=True)
            df_second_moment_interval.set_index('r', inplace=True)
            df_second_moment_interval.to_pickle(path=os.path.join(df_save_path, 'interval_second_moment.pkl'))

            df_eigen_interval.sort_values('r', inplace=True)
            df_eigen_interval.set_index('r', inplace=True)
            df_eigen_interval.to_pickle(path=os.path.join(df_save_path, 'eigen_interval.pkl'))

            break

        ref_values = data['ref_values']
        eigen_interval = data['eigen_interval']
        first_moment_interval = data['first_moment_interval']
        second_moment_interval = data['second_moment_interval']

        # add to df
        for df, interval in zip([df_first_moment_interval, df_second_moment_interval, df_eigen_interval],
                                [first_moment_interval, second_moment_interval, eigen_interval]):
            df.loc[-1] = [*ref_values,
                          interval.lower_bound,
                          interval.upper_bound,
                          interval.lower_bound_ci,
                          interval.upper_bound_ci,
                          interval.expectation,
                          interval.variance]
            df.index += 1
            df.sort_index()


def worker(r, ansatz, hamiltonian, backend, device, noise, samples, fidelity, exact, vqe_fn, use_grouping, nreps, q):
    # load vqe
    with open(vqe_fn.format(r=r), 'rb') as f:
        vqe = pickle.load(f)

    print(f'start interval calculations for r={r}...')
    # compute interval based on first moment
    first_moment_interval = ExpectationInterval(hamiltonian, ansatz)
    first_moment_interval.compute_interval(
        vqe.variables, backend, device, noise, samples, fidelity, use_grouping=use_grouping,
        use_second_moment=False, normalization_constants=None)
    print(f'finished first moment based interval for r={r}')

    # compute interval based on second moment
    constant_coeff = [ps.coeff.real for ps in hamiltonian.paulistrings if len(ps.naked()) == 0][0]
    normalization_constants = {
        'lower': constant_coeff - sum([abs(ps.coeff.real) for ps in hamiltonian.paulistrings if len(ps.naked()) > 0]),
        'upper': constant_coeff + sum([abs(ps.coeff.real) for ps in hamiltonian.paulistrings if len(ps.naked()) > 0])}

    second_moment_interval = ExpectationInterval(hamiltonian, ansatz)
    second_moment_interval.compute_interval(
        vqe.variables, backend, device, noise, samples, fidelity, use_grouping=use_grouping,
        use_second_moment=True, normalization_constants=normalization_constants, nreps=nreps)
    print(f'finished second moment based interval for r={r}')

    # compute eigenvalue interval
    eigen_interval = EigenvalueInterval(hamiltonian, ansatz)
    eigen_interval.compute_interval(vqe.variables, backend, device, noise, samples, fidelity, nreps=nreps)
    print(f'finished eigenvalue interval for r={r}')
    print(f'finished interval calculations for r={r}...')

    # put data in queue
    data = {'ref_values': [r, exact, vqe.energy, fidelity],
            'eigen_interval': eigen_interval,
            'first_moment_interval': first_moment_interval,
            'second_moment_interval': second_moment_interval
            }

    q.put(data)

    return data


def main(results_dir, molecule_name, use_grouping, nreps, samples=None):
    with open(os.path.join(results_dir, 'args.pkl'), 'rb') as f:
        loaded_args = pickle.load(f)

    if samples is not None and loaded_args.samples is not None:
        loaded_args.samples = samples

    if f'/{molecule_name.lower()}/' not in results_dir:
        raise ValueError(f'molecule_name {molecule_name.lower()} not in results_dir\n\t{results_dir} !')

    ansatz_name = loaded_args.ansatz
    result_df = pd.read_pickle(os.path.join(results_dir, 'energies.pkl'))
    bond_dists = result_df.index.to_list()

    molecule_initializer = get_molecule_initializer(geometry=geom_strings[molecule_name],
                                                    active_orbitals=active_orbitals[molecule_name])

    if loaded_args.gpu:
        try:
            jax_config.update("jax_platform_name", "gpu")
        except RuntimeError:
            jax_config.update("jax_platform_name", "cpu")
            print('WARNING! failed to set "jax_platform_name" to gpu; fallback to CPU')
    else:
        jax_config.update("jax_platform_name", "cpu")

    device = loaded_args.device
    backend = loaded_args.backend
    samples = loaded_args.samples

    if loaded_args.noise == 0:
        backend = 'qulacs'
        noise = None
        device = None
        samples = None
    elif loaded_args.noise == 1:
        noise = get_noise_model(1)
        device = None
    elif loaded_args.noise == 2:  # emulate device noise
        noise = 'device'
    else:
        raise NotImplementedError('noise model {} unknown'.format(loaded_args.noise))

    vqe_result_fn = os.path.join(results_dir, 'vqe_result_r={r}.pkl')
    chemistry_data_dir = DATA_DIR + f'{molecule_name}' + '_{r}'

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'intervals_out.txt'))

    # adjust to hcb
    transformation = 'JORDANWIGNER'

    if loaded_args.hcb:
        transformation = 'REORDEREDJORDANWIGNER'

        if 'hcb' not in ansatz_name.lower():
            ansatz_name = 'hcb-' + ansatz_name

    # classical calculations before multiprocessing
    bond_distances_final, exact_energies, fidelities = [], [], []
    molecules, ansatzes, hamiltonians = [], [], []

    for r in bond_dists:
        molecule = molecule_initializer(r=r, name=chemistry_data_dir,
                                        basis_set=loaded_args.basis_set,
                                        transformation=transformation,
                                        n_pno=N_PNO)

        if molecule is None:
            continue

        ansatz = make_ansatz(molecule, ansatz_name)
        hamiltonian = molecule.make_hamiltonian() if not loaded_args.hcb else molecule.make_hardcore_boson_hamiltonian()

        ansatzes.append(ansatz)
        hamiltonians.append(hamiltonian)

        bond_distances_final.append(r)
        molecules.append(molecule)

        exact_energies.append(result_df.loc[r]['exact'])
        fidelities.append(result_df.loc[r]['gs_fidelity'])

    print_summary(molecules[0], hamiltonians[0], ansatzes[0], use_grouping)

    del molecules

    num_processes = min(len(bond_distances_final), mp.cpu_count()) + 2

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_processes)

    # put listener to work first
    _ = pool.apply_async(listener, (q, results_dir))

    print(f'start_time\t: {timestamp_human()}')

    start_time = time.time()

    # fire off workers
    jobs = []
    for r, ansatz, hamiltonian, fidelity, exact in zip(
            bond_distances_final, ansatzes, hamiltonians, fidelities, exact_energies):
        job = pool.apply_async(
            worker,
            (r, ansatz, hamiltonian, backend, device, noise, samples, fidelity, exact, vqe_result_fn, use_grouping,
             nreps, q))

        jobs.append(job)

    # collect results
    for i, job in enumerate(jobs):
        job.get()

    q.put('kill')
    pool.close()
    pool.join()

    print(f'\nend_time\t: {timestamp_human()}')
    print(f'elapsed_time\t: {time.time() - start_time:.4f}s\n')

    print('\n=============================')
    print('\n\nFirst Moments Interval:\n')
    df = pd.read_pickle(os.path.join(results_dir, 'interval_first_moment.pkl'))
    print(tabulate(df, headers='keys', tablefmt='presto', floatfmt=".6f"))

    print('\n\nSecond Moments Lower Bound:\n')
    df = pd.read_pickle(os.path.join(results_dir, 'interval_second_moment.pkl'))
    print(tabulate(df, headers='keys', tablefmt='presto', floatfmt=".6f"))

    print('\n\nEigenvalue Interval:\n')
    df = pd.read_pickle(os.path.join(results_dir, 'eigen_interval.pkl'))
    print(tabulate(df, headers='keys', tablefmt='presto', floatfmt=".6f"))


if __name__ == '__main__':
    main(args.results_dir, args.molecule, args.use_grouping, args.reps, args.samples)
