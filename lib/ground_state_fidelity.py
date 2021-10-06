import numpy as np
import tequila as tq
from typing import List


def estimate_ground_state_fidelity(eigenvalues, eigenstates, ansatz, variables, backend, device=None, noise=None,
                                   samples=None, tol=2 * 1.5e-3, nreps=1) -> List[float]:
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

        return [max(fidelities)]

    fidelity_reps = []
    for _ in range(nreps):
        for i in ind_sorted:  # takes care of degenerate eigenvalues
            if abs(min_eigval - eigenvalues[i]) > tol:
                break

            deg_eigvals.append(eigenvalues[i])

            # compute fidelity
            exact_wfn = tq.QubitWaveFunction.from_array(eigenstates[:, i])
            exact_wfn = tq.paulis.Projector(wfn=exact_wfn)
            fidelity = tq.ExpectationValue(U=ansatz, H=exact_wfn)
            f = tq.simulate(objective=fidelity, variables=variables, backend=backend, device=device, noise=noise,
                            samples=samples)
            fidelities.append(f)

        fidelity_reps.append(max(fidelities))

    return fidelity_reps
