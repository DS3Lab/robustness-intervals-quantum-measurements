import sys
from datetime import datetime as dt
import numpy as np
import os

import tequila as tq


def timestamp():
    return dt.now().strftime('%y%m%d_%H%M%S')


def timestamp_human():
    return dt.now().strftime('%H:%M:%S')


def estimate_fidelity(eigenvalues, eigenstates, ansatz, variables, backend, device=None, noise=None, samples=None,
                      tol=2 * 1.5e-3):
    fidelities = []

    # sort eigenvalues
    ind_sorted = np.argsort(eigenvalues)
    min_eigval = eigenvalues[ind_sorted[0]]
    deg_eigvals = []

    if noise is None:
        for i in ind_sorted:  # takes care of degenerate eigenvalues
            if abs(min_eigval - eigenvalues[i]) > tol:
                break

            deg_eigvals.append(eigenvalues[i])

            # compute fidelity
            exact_wfn = tq.QubitWaveFunction.from_array(eigenstates[:, i])
            ansatz_wfn = tq.simulate(ansatz, backend=backend, variables=variables)
            f = abs(exact_wfn.inner(ansatz_wfn)) ** 2

            fidelities.append(f)

        # if len(fidelities) > 1:
        #     print(f'Warning! found {len(fidelities)} degenerate eigenvalues within tol = {tol};' +
        #           f'\n\teigenvalues:\t{deg_eigvals}\n\tfidelities:\t{fidelities}')

        return max(fidelities)

    for i in ind_sorted:  # takes care of degenerate eigenvalues
        if abs(min_eigval - eigenvalues[i]) > tol:
            break

        deg_eigvals.append(eigenvalues[i])

        # compute fidelity
        exact_wfn = tq.QubitWaveFunction.from_array(eigenstates[:, i])
        exact_wfn = tq.paulis.Projector(wfn=exact_wfn)
        fidelity = tq.ExpectationValue(U=ansatz, H=exact_wfn)
        f = tq.simulate(objective=fidelity, variables=variables, backend=backend, device=device, noise=noise,  # noqa
                        samples=samples)

        fidelities.append(f)

    # if len(fidelities) > 1:
    #     print(f'Warning! found {len(fidelities)} degenerate eigenvalues within tol = {tol};' +
    #           f'\n\teigenvalues:\t{deg_eigvals}\n\tfidelities:\t{fidelities}')

    return max(fidelities)


class Logger:
    def __init__(self, print_fp=None):
        self.terminal = sys.stdout
        self.log_file = "logfile.txt" if print_fp is None else print_fp
        self.encoding = sys.stdout.encoding

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as log:
            log.write(message)

    def flush(self):
        pass


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
