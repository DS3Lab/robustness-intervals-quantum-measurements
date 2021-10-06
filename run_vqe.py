import argparse
import _pickle as pickle
from collections import namedtuple
import copy
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import sys
import time

import tequila as tq

from constants import ROOT_DIR, DATA_DIR
from lib.make_ansatz import make_ansatz
from lib.molecule_initializer import get_molecule_initializer
from lib.helpers import print_summary, timestamp_human, Logger
from lib.noise_model import get_noise_model

parser = argparse.ArgumentParser()
parser.add_argument("--molecule", type=str, choices=['h2', 'lih', 'beh2'])
parser.add_argument("--ansatz", type=str, required=True, choices=['upccgsd', 'spa-gas', 'spa-gs', 'spa-s', 'spa'])
parser.add_argument("--basis-set", "-bs", type=str, default=None)
parser.add_argument("--noise", type=str, default=None, choices=[None, 'bitflip-depol', 'device'])
parser.add_argument("--error-rate", type=float, default=1e-2)
parser.add_argument("--samples", "-s", type=int, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--results-dir", type=str, default=os.path.join(ROOT_DIR, "results/"))
parser.add_argument("--optimizer", type=str, default='COBYLA')
parser.add_argument("--backend", type=str, default="qulacs")
parser.add_argument("--restarts", type=int, default=1)
args = parser.parse_args()

n_pno = None
transformation = 'JORDANWIGNER'

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

columns = ["r", "fci", "mp2", "ccsd", "vqe"]


def main():
    molecule_name = args.molecule.lower()
    ansatz_name = args.ansatz

    if args.basis_set is None:
        bond_dists = filtered_dists(DATA_DIR, args.molecule.lower())
    else:
        bond_dists = list(np.arange(start=0.4, stop=5.0, step=0.2).round(2))

    molecule_initializer = get_molecule_initializer(geometry=geom_strings[molecule_name],
                                                    active_orbitals=active_orbitals[molecule_name])

    noise = get_noise_model(error_rate=args.error_rate, noise_type=args.noise)

    if args.noise is None:
        backend = 'qulacs'
        device = None
        samples = None
        args.error_rate = 0
    elif args.noise == 'bitflip-depol':
        backend = args.backend
        samples = args.samples
        device = None
    elif args.noise == 'device':  # emulate device noise
        backend = args.backend
        device = args.device
        samples = args.samples
    else:
        raise NotImplementedError('noise model {} unknown'.format(args.noise))

    # build dir structure
    save_dir = os.path.join(args.results_dir, f"./{molecule_name}/")
    save_dir = os.path.join(save_dir, f"{'basis-set-free' if args.basis_set is None else args.basis_set}/")
    save_dir = os.path.join(save_dir, f"noise={args.noise if device is None else device}_error-rate={args.error_rate}/")
    save_dir = os.path.join(save_dir, f"{timestamp_human()}/".replace(':', '-').replace(' ', '_'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_file = os.path.join(save_dir, 'energies.pkl')
    vqe_result_fn = save_dir + 'vqe_result_r={r}.pkl'
    args_fn = os.path.join(save_dir, 'args.pkl')

    with open(args_fn, 'wb') as f:
        pickle.dump(args, f)
        print(f'saved args to {args_fn}')

    chemistry_data_dir = DATA_DIR + f'{molecule_name}' + '_{r}'

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(save_dir, 'vqe_out.txt'))

    # classical calculations before multiprocessing
    bond_distances_final = []
    fci_vals, mp2_vals, ccsd_vals, hf_vals = [], [], [], []
    molecules = []
    ansatzes, hamiltonians = [], []

    for r in bond_dists:
        molecule = molecule_initializer(
            r=r, name=chemistry_data_dir, basis_set=args.basis_set, transformation=transformation, n_pno=n_pno)

        if molecule is None:
            continue

        # make classical computations
        hf_vals.append(compute_energy_classical('hf', molecule))
        fci_vals.append(compute_energy_classical('fci', molecule))
        mp2_vals.append(compute_energy_classical('mp2', molecule))
        ccsd_vals.append(compute_energy_classical('ccsd', molecule))

        # build ansatz
        ansatz = make_ansatz(molecule, ansatz_name)
        hamiltonian = molecule.make_hamiltonian()

        ansatzes.append(ansatz)
        hamiltonians.append(hamiltonian)

        bond_distances_final.append(r)
        molecules.append(molecule)

    print_summary(molecules[0], hamiltonians[0], ansatzes[0], ansatz_name, None)

    del molecules

    num_processes = min(len(bond_dists), mp.cpu_count()) + 2

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_processes)

    # put listener to work first
    _ = pool.apply_async(listener, (q, results_file))

    print(f'start_time\t: {timestamp_human()}')

    # print progress header
    print("\n{:^25} | ".format("timestamp") + " | ".join(["{:^12}".format(v) for v in columns]))
    print("-" * 26 + "+" + "+".join(["-" * 14 for _ in range(len(columns))]))

    start_time = time.time()

    # fire off workers
    jobs = []
    for r, ansatz, hamiltonian, fci, mp2, ccsd in zip(bond_distances_final, ansatzes, hamiltonians, fci_vals, mp2_vals,
                                                      ccsd_vals):
        job = pool.apply_async(
            worker,
            (r, ansatz, hamiltonian, args.optimizer, backend, device, noise, samples, fci, mp2, ccsd,
             args.restarts, vqe_result_fn, q))

        jobs.append(job)

    # collect results
    for i, job in enumerate(jobs):
        job.get()

    q.put('kill')
    pool.close()
    pool.join()

    print(f'\nend_time\t: {timestamp_human()}')
    print(f'elapsed_time\t: {time.time() - start_time:.4f}s')


def listener(q, df_save_as):
    df = pd.DataFrame(columns=columns)

    while 1:
        m = q.get()

        if m == "kill":
            df.sort_values('r', inplace=True)
            df.set_index('r', inplace=True)
            df.to_pickle(path=df_save_as)
            break

        print("{:^14} | ".format("[" + timestamp_human() + "]") + " | ".join(["{:^12}".format(round(v, 6)) for v in m]))

        # add to df
        df.loc[-1] = m
        df.index += 1
        df.sort_index()


def worker(r, ansatz, hamiltonian, optimizer, backend, device, noise, samples, fci, mp2, ccsd, restarts, vqe_fn, q):
    # run vqe
    objective = tq.ExpectationValue(U=ansatz, H=hamiltonian, optimize_measurements=False)

    Result = namedtuple('result', 'energy')
    result = Result(energy=np.inf)

    # restart optimization n_reps times
    for _ in range(restarts):
        init_vals = {k: np.random.normal(loc=0, scale=np.pi / 4.0) for k in objective.extract_variables()}
        temp_result = tq.minimize(objective, method=optimizer, initial_values=init_vals, silent=True,
                                  backend=backend, device=device, noise=noise, samples=samples)

        if temp_result.energy <= result.energy:
            result = copy.deepcopy(temp_result)

    # save SciPyResults
    with open(vqe_fn.format(r=r), 'wb') as f:
        pickle.dump(result, f)

    # put data in queue
    q.put([r, fci, mp2, ccsd, result.energy])


def filtered_dists(data_dir, mol_str):
    """
    returns the list of bond distances corresponding to molecule files in data_dir
    """
    dists = []
    for fn in os.listdir(data_dir):
        if mol_str in fn and "_htensor.npy" in fn:
            if len(fn.split(mol_str)[0]) == 0:
                dist = float(fn.split(f"{mol_str}_")[-1].split("_htensor.npy")[0])
                dists.append(dist)

    return dists


def compute_energy_classical(method, molecule):
    try:
        return molecule.compute_energy(method)
    except Exception as e:
        print(f'caught exception! {e}')
        return np.nan


if __name__ == '__main__':
    main()
