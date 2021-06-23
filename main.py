import argparse
import numpy as np
import os
import multiprocessing as mp
from psi4 import SCFConvergenceError

import tequila as tq

from constants import ROOT_DIR, DATA_DIR

from simulation import run_simulation
from lib.helpers import filtered_dists

parser = argparse.ArgumentParser()
parser.add_argument("--molecule", type=str, choices=['h2', 'lih', 'beh2'])
parser.add_argument("--ansatz", type=str, choices=['upccgsd', 'spa-gas', 'spa-gs', 'spa-s', 'spa'])
parser.add_argument("--basis_set", "-bs", type=str, default=None)
parser.add_argument("--hcb", action="store_true")
parser.add_argument("--noise", type=int, default=0, choices=[0, 1, 2])
parser.add_argument("--samples", "-s", type=int, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--results_dir", type=str, default=os.path.join(ROOT_DIR, "results/"))
parser.add_argument("--backend", type=str, default="qulacs")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--reps", type=int, default=1)
parser.add_argument("--rand_dir", action="store_true")
parser.add_argument("--use_grouping", type=int, default=1, choices=[0,1])
args = parser.parse_args()
print("use_grouping=", args.use_grouping)
TRANSFORMATION = 'JORDANWIGNER'
N_PNO = None
OPTIMIZER = 'BFGS'


def initialize_molecule(r, geometry, name, basis_set, active_orbitals, transformation, n_pno):
    try:
        if basis_set is None:
            return tq.Molecule(geometry=geometry.format(r=r),
                               basis_set=basis_set,
                               transformation=transformation,
                               name=name.format(r=r),
                               n_pno=n_pno)
        else:
            return tq.Molecule(geometry=geometry.format(r=r),
                               basis_set=basis_set,
                               active_orbitals=active_orbitals,
                               transformation=transformation,
                               name=name.format(r=r),
                               n_pno=n_pno)

    except SCFConvergenceError:
        print('WARNING! could not intialize molecule with bond_distance {r}'.format(r=r))
        return None


def main():
    if args.molecule == 'h2':
        active_orbitals = None

        if args.basis_set is None:
            bond_dists = filtered_dists(DATA_DIR, 'h2')
        else:
            bond_dists = list(np.arange(start=0.4, stop=5.0, step=0.2).round(2))

        def init_mol(r, name, basis_set, transformation, n_pno):
            return initialize_molecule(r=r, geometry='H .0 .0 .0\nH .0 .0 {r}',
                                       name=name,
                                       basis_set=basis_set,
                                       active_orbitals=active_orbitals,
                                       transformation=transformation,
                                       n_pno=n_pno)

    elif args.molecule == 'lih':
        active_orbitals = None

        if args.basis_set is None:
            bond_dists = filtered_dists(DATA_DIR, 'lih')
        else:
            bond_dists = list(np.arange(start=0.7, stop=5.0, step=0.25).round(2)) + [1.6]

        def init_mol(r, name, basis_set, transformation, n_pno):
            return initialize_molecule(r, 'H .0 .0 .0\nLi .0 .0 {r}', name, basis_set, active_orbitals, transformation,
                                       n_pno)

    elif args.molecule == 'beh2':
        active_orbitals = None

        if args.basis_set is None:
            bond_dists = filtered_dists(DATA_DIR, 'beh2')
        else:
            bond_dists = list(np.arange(start=0.8, stop=5.0, step=0.2).round(2))

        def init_mol(r, name, basis_set, transformation, n_pno):
            return initialize_molecule(r, 'Be .0 .0 .0\nH .0 .0 {r}\nH .0 .0 -{r}', name, basis_set, active_orbitals,
                                       transformation, n_pno)

    else:
        raise NotImplementedError('molecule {} not implemented!')

    # qulacs backend uses multithreading
    if args.backend in [None, "None", "qulacs"]:
        omp_threads = os.getenv('OMP_NUM_THREADS')
        if omp_threads is None:
            # this means qulacs will grab all threads
            omp_threads = mp.cpu_count()
        else:
            omp_threads = int(omp_threads)
        num_processes = max(2,mp.cpu_count() - omp_threads)
    else:
        num_processes = 2
    
    bond_dists = sorted(bond_dists)
    print(bond_dists)
    print(num_processes)
    run_simulation(molecule_name=args.molecule,
                   initialize_molecule=init_mol,
                   optimizer=OPTIMIZER,
                   bond_distances=bond_dists,
                   ansatz_name=args.ansatz,
                   hcb=args.hcb,
                   use_gpu=args.gpu,
                   backend=args.backend,
                   device=args.device,
                   noise_id=args.noise,
                   samples=args.samples,
                   results_dir=args.results_dir,
                   basis_set=args.basis_set,
                   transformation=TRANSFORMATION,
                   n_pno=N_PNO,
                   n_reps=args.reps,
                   random_dir=args.rand_dir,
                   num_processes=num_processes,
                   use_grouping=args.use_grouping)


if __name__ == '__main__':
    main()
