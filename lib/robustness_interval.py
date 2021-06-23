import numpy as np
import warnings
from .helpers import make_paulicliques

import tequila as tq


def g(nu, f):
    if f < nu:
        return 0
    else:
        return nu * (1.0 - 2.0 * f) + f - 2.0 * np.sqrt(nu * (1.0 - nu) * f * (1.0 - f))


class RobustnessInterval:
    def __init__(self, hamiltonian, ansatz, variables, fidelity: float = None, confidence_level: float = 0.01):
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._variables = variables
        self._fidelity = fidelity
        self._confidence_level = confidence_level

        if self._fidelity < 0:
            warnings.warn("fidelity={}; setting to 0 manually.".format(fidelity), RuntimeWarning)
            self._fidelity = 0.0

        self._lower = 0.0
        self._upper = 0.0
        self._expectation = 0.0

    def compute_interval(self, backend, device, noise, samples, use_grouping):

        # extract pauli terms
        if not use_grouping:
            pauli_strings = [ps.naked() for ps in self._hamiltonian.paulistrings]
            pauli_coeffs = [ps.coeff.real for ps in self._hamiltonian.paulistrings]
        else:
            paulicliques = make_paulicliques(self._hamiltonian)
            pauli_strings = [ps.naked() for ps in paulicliques]
            pauli_coeffs = [ps.coeff.real for ps in paulicliques]
        # finite sampling error (hoeffding)
        num_paulis = len(pauli_coeffs)
        if samples is None:
            fs_err = 0.0
        else:
            fs_err = np.sqrt(-np.log(0.5 * (1 - (1 - self._confidence_level) ** (1.0 / num_paulis))) / (2 * samples))

        for p_str, p_coeff in zip(pauli_strings, pauli_coeffs):
            # eval pauli
            if str(p_str) == 'I' or len(p_str) == 0:
                pauli_expec = 1.0
            else:
                if hasattr(p_str, "H"):
                    observable = p_str.H
                else:
                    observable = tq.QubitHamiltonian.from_paulistrings([p_str])
                if hasattr(p_str, "U"):
                    U = self._ansatz + p_str.U
                else:
                    U = self._ansatz
                objective = tq.ExpectationValue(U=U, H=observable)
                pauli_expec = tq.simulate(objective, variables=self._variables, samples=samples, backend=backend, device=device, noise=noise)

            # pauli bounds
            min_eigenvalue=-1.0
            max_eigenvalue=1.0
            if hasattr(p_str, "compute_eigenvalues"):
                ev = p_str.compute_eigenvalues()
                min_eigenvalue=min(ev)
                max_eigenvalue=max(ev)
            pauli_expec = np.clip(pauli_expec, min_eigenvalue, max_eigenvalue)
                
            lower_bound, upper_bound = self._compute_interval_single(p_str, pauli_expec, fs_err, self._fidelity, min_eigenvalue, max_eigenvalue)

            # update total expectation
            self._expectation += p_coeff * pauli_expec
            self._lower += p_coeff * lower_bound if p_coeff > 0 else p_coeff * upper_bound
            self._upper += p_coeff * upper_bound if p_coeff > 0 else p_coeff * lower_bound

    @staticmethod
    def _compute_interval_single(pauli_string, pauli_expecation, fs_err, fidelity, min_eigenvalue, max_eigenvalue):
        if str(pauli_string) == 'I' or len(pauli_string) == 0:
            return 1.0, 1.0
        c = max_eigenvalue-min_eigenvalue
        d = min_eigenvalue
        x = (pauli_expecation+fs_err-d)/c # @Maurice: Maybe check with the fs_err here
        
        lower_bound = g(1.0-x, fidelity)
        upper_bound = 1.0-g(x,fidelity)
        lower_bound = c*lower_bound + d
        upper_bound = c*upper_bound + d

        return lower_bound, upper_bound

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @property
    def ansatz(self):
        return self._ansatz

    @property
    def variables(self):
        return self._variables

    @property
    def fidelity(self):
        return self._fidelity

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def expectation(self):
        return self._expectation
