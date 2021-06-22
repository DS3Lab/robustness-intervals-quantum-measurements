import multiprocessing as mp
import time

import copy
from collections import namedtuple
import csv
import numpy as np
import os
from jax.config import config as jax_config
import sys

import tequila as tq

from lib.robustness_interval import RobustnessInterval
from lib.helpers import timestamp_human, timestamp, estimate_fidelity, Logger
from lib.noise_models import get_noise_model
from constants import DATA_DIR

ALPHA = 1e-2


def listener(fn, q):
    while 1:
        m = q.get()

        if m == "kill":
            break

        print("{:^14} | ".format("[" + timestamp_human() + "]") + " | ".join(["{:^12}".format(round(v, 6)) for v in m]))

        with open(fn, "a") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow(m)
            f.flush()


def worker(r, ansatz, hamiltonian, optimizer, backend, device, noise, samples, fci, mp2, ccsd, n_reps, q):
    # TODO: save VQE params

    # run vqe
    objective = tq.ExpectationValue(U=ansatz, H=hamiltonian, optimize_measurements=True)

    Result = namedtuple('result', 'energy')
    result = Result(energy=np.inf)

    # restart optimization n_reps times
    for _ in range(n_reps):
        init_vals = {k: np.random.normal(loc=0, scale=np.pi / 4.0) for k in objective.extract_variables()}
        temp_result = tq.minimize(objective, method=optimizer, initial_values=init_vals, silent=True,
                                  backend=backend, device=device, noise=noise, samples=samples)

        if temp_result.energy <= result.energy:
            result = copy.deepcopy(temp_result)

    # compute exact solution
    hamiltonian_matrix = hamiltonian.to_matrix()  # noqa
    eigenvalues, eigenstates = np.linalg.eigh(hamiltonian_matrix)
    exact = min(eigenvalues)

    # compute fidelity
    fidelity = estimate_fidelity(eigenvalues, eigenstates, ansatz, result.variables, backend, device, noise, samples)

    # compute robustness interval
    ri = RobustnessInterval(hamiltonian, ansatz, result.variables, fidelity, ALPHA)
    ri.compute_interval(backend, device, noise, samples)

    # compute trivial bounds: ± \sum_i |\omega_i| + const.
    constant_coeff = [ps.coeff.real for ps in hamiltonian.paulistrings if len(ps.naked()) == 0][0]
    trivial_bound0 = np.sum(np.abs([ps.coeff.real for ps in hamiltonian.paulistrings if len(ps) != 0]))
    trivial_bounds = constant_coeff - trivial_bound0, constant_coeff + trivial_bound0

    data = [r, exact, fci, mp2, ccsd, ri.expectation, ri.lower, ri.upper, fidelity, *trivial_bounds]

    q.put(data)

    return data


def compute_energy_classical(method, molecule):
    try:
        return molecule.compute_energy(method)
    except Exception:  # noqa
        return np.nan


def build_ansatz(molecule, name):
    name = name.lower()

    if name == 'upccgsd':
        return molecule.make_upccgsd_ansatz(name='upccgsd')

    if 'spa' in name:
        ansatz = molecule.make_upccgsd_ansatz(name='SPA')
        if 'spa-gas' in name:
            ansatz += molecule.make_upccgsd_ansatz(name='GAS', include_reference=False)

        if 'spa-gs' in name:
            ansatz += molecule.make_upccgsd_ansatz(name='GS', include_reference=False)

        if 'spa-s' in name:
            ansatz += molecule.make_upccgsd_ansatz(name='S', include_reference=False)

        return ansatz

    raise NotImplementedError(f'Ansatz {name} not known')


def summarize_molecule(molecule, hamiltonian, ansatz):
    print(f"""---- molecule summary ----
molecule: {molecule}
    
n_orbitals\t: {molecule.n_orbitals}
n_electrons\t: {molecule.n_electrons}
    
Hamiltonian:
num_terms\t: {len(hamiltonian)}
num_qubits\t: {hamiltonian.n_qubits}
    
Ansatz:
num_params\t: {len(ansatz.extract_variables())}
    """)


def run_simulation(molecule_name, initialize_molecule, optimizer, bond_distances, ansatz_name, hcb, use_gpu, backend,
                   device, noise_id, samples, results_dir, basis_set, transformation, n_pno, n_reps=1,
                   random_dir=False, num_processes=None):
    if use_gpu:
        try:
            jax_config.update("jax_platform_name", "gpu")
        except RuntimeError:
            jax_config.update("jax_platform_name", "cpu")
            print('WARNING! failed to set "jax_platform_name" to gpu; fallback to CPU')
    else:
        jax_config.update("jax_platform_name", "cpu")

    start_time = time.time()
    molecule_name = molecule_name.lower()

    if noise_id == 0:
        backend = 'qulacs'
        noise = None
        device = None
        samples = None
    elif noise_id == 1:
        noise = get_noise_model(1)
        device = None
    elif noise_id == 2:  # emulate device noise
        noise = 'device'
    else:
        raise NotImplementedError('noise model {} unknown'.format(noise_id))

    # build dir structure
    save_dir = os.path.join(results_dir, f"./{molecule_name}/")
    save_dir = os.path.join(save_dir, f"{'basis-set-free' if basis_set is None else basis_set}/")
    save_dir = os.path.join(save_dir, f"hcb={hcb}/{ansatz_name}/")
    save_dir = os.path.join(save_dir, f"noise={noise_id if device is None else device}")

    if random_dir:
        save_dir = os.path.join(save_dir, f"{timestamp()}")

    csv_fp = os.path.join(save_dir, "results.csv")
    data_dir = DATA_DIR + f'{molecule_name}' + '_{r}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # initialize csv
    with open(csv_fp, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = ["r", "exact", "fci", "mp2", "ccsd", "vqe", "lower_bound", "upper_bound", "fidelity", "lower_bound0",
                  "upper_bound0"]
        csv_writer.writerow(header)

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(save_dir, 'out.txt'))

    # adjust to hcb
    if hcb:
        if transformation.lower() == 'jordanwigner':
            transformation = 'REORDEREDJORDANWIGNER'
        else:
            raise NotImplementedError

        if 'hcb' not in ansatz_name.lower():
            ansatz_name = 'hcb-' + ansatz_name

    # make psi4 calculations before multiprocessing
    bond_distances_final = []
    fci_vals, mp2_vals, ccsd_vals = [], [], []
    molecules = []
    ansatzes, hamiltonians = [], []

    for r in bond_distances:
        molecule = initialize_molecule(r=r, name=data_dir, basis_set=basis_set, transformation=transformation,
                                       n_pno=n_pno)

        if molecule is None:
            continue

        # make classical computations
        fci_vals.append(compute_energy_classical('fci', molecule))  # noqa
        mp2_vals.append(compute_energy_classical('mp2', molecule))  # noqa
        ccsd_vals.append(compute_energy_classical('ccsd', molecule))  # noqa

        ansatz = build_ansatz(molecule, ansatz_name)
        hamiltonian = molecule.make_hamiltonian() if not hcb else molecule.make_hardcore_boson_hamiltonian()

        ansatzes.append(ansatz)
        hamiltonians.append(hamiltonian)

        bond_distances_final.append(r)
        molecules.append(molecule)

    summarize_molecule(molecules[0], hamiltonians[0], ansatzes[0])

    del molecules

    if num_processes is None:
        num_processes = min(len(bond_distances), mp.cpu_count()) + 2

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_processes)

    # put listener to work first
    _ = pool.apply_async(listener, (csv_fp, q))

    # print progress header
    print("\n{:^14} | ".format("time") + " | ".join(["{:^12}".format(v) for v in header]))
    print("-" * 15 + "+" + "+".join(["-" * 14 for _ in range(len(header))]))

    # fire off workers
    jobs = []
    for r, ansatz, hamiltonian, fci, mp2, ccsd in zip(bond_distances_final, ansatzes, hamiltonians,
                                                      fci_vals, mp2_vals, ccsd_vals):
        job = pool.apply_async(
            worker,
            (r, ansatz, hamiltonian, optimizer, backend, device, noise, samples, fci, mp2, ccsd, n_reps, q)
        )

        jobs.append(job)

    # collect results
    for i, job in enumerate(jobs):
        job.get()

    q.put('kill')
    pool.close()
    pool.join()

    print('total time elapsed: {:.6f}'.format(time.time() - start_time))
