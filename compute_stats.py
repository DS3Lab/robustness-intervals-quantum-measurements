import argparse
import _pickle as pickle
import multiprocessing as mp
import os
import pandas as pd
import sys
import numpy as np
import time

import tequila as tq

from constants import DATA_DIR
from lib.make_ansatz import make_ansatz
from lib.molecule_initializer import get_molecule_initializer
from lib.pauliclique import make_paulicliques
from lib.ground_state_fidelity import estimate_ground_state_fidelity
from lib.helpers import print_summary, timestamp_human, Logger
from lib.noise_model import get_noise_model

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, required=True, help='dir from which to load results')
parser.add_argument("--which", type=str, required=False, default='all',
                    choices=['pauli', 'paulicliques', 'hamiltonian', 'all'])
parser.add_argument("--samples", "-s", type=int, default=None)
parser.add_argument("--reps", type=int, default=20)
args = parser.parse_args()

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

n_pno = None
transformation = 'JORDANWIGNER'

columns = ["r", "E0", "E1", "gs_fidelity_reps",
           "hamiltonian_expectations", "hamiltonian_variances",
           "pauli_strings", "pauli_coeffs", "pauli_expectations", "pauli_eigenvalues",
           "paulicliques", "paulicliques_expectations", "paulicliques_variances", "paulicliques_eigenvalues",
           "nreps", "samples"]


def main(results_dir, nreps, which_stats, samples=None):
    with open(os.path.join(results_dir, 'args.pkl'), 'rb') as f:
        loaded_args = pickle.load(f)

    if samples is not None and loaded_args.samples is not None:
        loaded_args.samples = samples

    molecule_name = loaded_args.molecule
    ansatz_name = loaded_args.ansatz
    energies_df = pd.read_pickle(os.path.join(results_dir, 'energies.pkl'))
    bond_dists = energies_df.index.to_list()

    molecule_initializer = get_molecule_initializer(geometry=geom_strings[molecule_name],
                                                    active_orbitals=active_orbitals[molecule_name])

    noise = get_noise_model(1e-2, noise_type={0: None, 1: 'bitflip-depol'}[loaded_args.noise])

    if loaded_args.noise is None or loaded_args.noise == 0:
        backend = 'qulacs'
        device = None
        samples = None
        loaded_args.error_rate = 0
    elif loaded_args.noise == 'bitflip-depol' or loaded_args.noise == 1:
        backend = loaded_args.backend
        samples = loaded_args.samples
        device = None
    elif loaded_args.noise == 'device':  # emulate device noise
        backend = loaded_args.backend
        device = loaded_args.device
        samples = args.samples
    else:
        raise NotImplementedError('noise model {} unknown'.format(loaded_args.noise))

    vqe_result_fn = os.path.join(results_dir, 'vqe_result_r={r}.pkl')
    chemistry_data_dir = DATA_DIR + f'{molecule_name}' + '_{r}'

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, f'{which_stats}_stats_out.txt'))

    print('\n----------------------\n')
    for k, v in args.__dict__.items():
        print('{0:20}: {1}'.format(k, v))
    print('\n----------------------\n')

    # classical calculations before multiprocessing
    bond_distances_final = []
    molecules, ansatzes, hamiltonians = [], [], []

    for r in bond_dists:
        molecule = molecule_initializer(r=r, name=chemistry_data_dir,
                                        basis_set=loaded_args.basis_set,
                                        transformation=transformation,
                                        n_pno=n_pno)
        if molecule is None:
            continue

        ansatz = make_ansatz(molecule, ansatz_name)
        hamiltonian = molecule.make_hamiltonian()

        ansatzes.append(ansatz)
        hamiltonians.append(hamiltonian)

        bond_distances_final.append(r)
        molecules.append(molecule)

    print_summary(molecules[0], hamiltonians[0], ansatzes[0], ansatz_name, None)

    del molecules

    num_processes = min(len(bond_distances_final), mp.cpu_count()) + 2

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_processes)

    # put listener to work first
    _ = pool.apply_async(listener, (q, results_dir, which_stats))

    # set nreps to 1 if no sampling
    if samples is None and nreps > 1:
        print('setting nreps to 1 since no sampling')
        nreps = 1

    print(f'start_time\t: {timestamp_human()}')

    start_time = time.time()

    # fire off workers
    jobs = []
    for r, ansatz, hamiltonian in zip(bond_distances_final, ansatzes, hamiltonians):
        job = pool.apply_async(
            worker, (r, ansatz, hamiltonian, backend, device, noise, samples, vqe_result_fn, nreps, which_stats, q))
        jobs.append(job)

    # collect results
    for i, job in enumerate(jobs):
        job.get()

    q.put('kill')
    pool.close()
    pool.join()

    print(f'\nend_time\t: {timestamp_human()}')
    print(f'elapsed_time\t: {time.time() - start_time:.4f}s\n')


def listener(q, df_save_path, which_stats):
    df = pd.DataFrame(columns=columns)

    while 1:
        data = q.get()

        if data == "kill":
            df.sort_values('r', inplace=True)
            df.set_index('r', inplace=True)

            print('saving data as', os.path.join(df_save_path, f'{which_stats}_statistics.pkl'))
            df.to_pickle(path=os.path.join(df_save_path, f'{which_stats}_statistics.pkl'))
            break

        try:
            df = df.append(data, ignore_index=True)
        except Exception as e:
            print('exception occured!', e)

        df.sort_index()


def worker(r, ansatz, hamiltonian, backend, device, noise, samples, vqe_fn, nreps, which_stats, q):
    # load vqe
    with open(vqe_fn.format(r=r), 'rb') as f:
        vqe = pickle.load(f)

    print(f'\n[{timestamp_human()}] start computing statistics for r={r}')

    # compute exact solution and spectral gap
    hamiltonian_matrix = hamiltonian.to_matrix()
    eigenvalues, eigenstates = np.linalg.eigh(hamiltonian_matrix)
    min_eigval = min(eigenvalues)  # ground state energy
    second_eigval = min(eigenvalues[eigenvalues > min_eigval])  # energy of first excited state

    # compute ground state fidelity
    ground_state_fidelity_reps = estimate_ground_state_fidelity(eigenvalues, eigenstates, ansatz, vqe.variables,
                                                                backend, device, noise, samples, nreps=nreps)

    data_basic = {'r': r,
                  'E0': min_eigval,
                  'E1': second_eigval,
                  'gs_fidelity_reps': ground_state_fidelity_reps,
                  'nreps': nreps,
                  'samples': samples}

    # compute stats for hamiltonian
    data_hamiltonian = {}
    if which_stats.lower() in ['hamiltonian', 'all']:
        hamiltonian_expectations, hamiltonian_variances = [], []

        for _ in range(nreps):
            # compute expectation
            e = tq.simulate(tq.ExpectationValue(U=ansatz, H=hamiltonian),
                            variables=vqe.variables, samples=samples, backend=backend, device=device,
                            noise=noise)

            # compute variance
            v = tq.simulate(tq.ExpectationValue(U=ansatz, H=(hamiltonian - e) ** 2),
                            variables=vqe.variables, samples=samples, backend=backend, device=device, noise=noise)

            hamiltonian_expectations.append(e)
            hamiltonian_variances.append(v)

        data_hamiltonian = {'hamiltonian_expectations': hamiltonian_expectations,
                            'hamiltonian_variances': hamiltonian_variances}

    # compute stats for individual pauli terms
    data_paulis = {}
    if which_stats.lower() in ['pauli', 'all']:
        pauli_strings = [ps.naked() for ps in hamiltonian.paulistrings]
        pauli_coeffs = [ps.coeff.real for ps in hamiltonian.paulistrings]

        pauli_expectations, pauli_variances = [], []

        for _ in range(nreps):
            # compute expectations
            ee = [tq.simulate(tq.ExpectationValue(H=tq.QubitHamiltonian.from_paulistrings([p_str]), U=ansatz),
                              variables=vqe.variables, samples=samples, backend=backend, device=device, noise=noise)
                  for p_str in pauli_strings]

            # compute variances
            vv = [
                tq.simulate(tq.ExpectationValue(U=ansatz, H=(tq.QubitHamiltonian.from_paulistrings([p_str]) - e) ** 2),
                            variables=vqe.variables, samples=samples, backend=backend, device=device, noise=noise)
                for p_str, e in zip(pauli_strings, ee)
            ]

            pauli_expectations.append(ee)
            pauli_variances.append(vv)

        pauli_eigenvalues = [(-1.0, 1.0) for _ in hamiltonian.paulistrings]

        data_paulis = {'pauli_strings': pauli_strings,
                       'pauli_coeffs': pauli_coeffs,
                       'pauli_expectations': pauli_expectations,
                       'pauli_variances': pauli_variances,
                       'pauli_eigenvalues': pauli_eigenvalues}

    # compute stats for pauli cliques
    data_paulicliques = {}
    if which_stats.lower() in ['paulicliques', 'all']:
        # compute pauli expectations w/ grouping
        paulicliques = make_paulicliques(hamiltonian)
        objectives = [tq.ExpectationValue(U=ansatz + clique.U, H=clique.H) for clique in paulicliques]

        paulicliques_expectations, paulicliques_variances = [], []
        for _ in range(nreps):
            # compute expectations
            ee = [tq.simulate(o, variables=vqe.variables, samples=samples, backend=backend, device=device, noise=noise)
                  for o in objectives]

            # compute variances
            vv = [
                tq.simulate(tq.ExpectationValue(U=ansatz + clique.U, H=(clique.H - e) ** 2),
                            variables=vqe.variables, samples=samples, backend=backend, device=device, noise=noise)
                for clique, e in zip(paulicliques, ee)
            ]

            paulicliques_expectations.append(ee)
            paulicliques_variances.append(vv)

        pauliclique_eigenvalues = [clique.compute_eigenvalues() for clique in paulicliques]

        data_paulicliques = {'paulicliques': paulicliques,
                             'paulicliques_coeffs': np.ones_like(paulicliques_expectations[0]).tolist(),
                             'paulicliques_expectations': paulicliques_expectations,
                             'paulicliques_variances': paulicliques_variances,
                             'paulicliques_eigenvalues': pauliclique_eigenvalues}

    print(f'\n[{timestamp_human()}] finished computing statistics for r={r}')

    # put data in queue
    data = {**data_basic, **data_hamiltonian, **data_paulis, **data_paulicliques}
    data.update({k: np.nan for k in columns if k not in data})

    q.put(data)
    return


if __name__ == '__main__':
    main(args.results_dir, args.reps, args.which, args.samples)
