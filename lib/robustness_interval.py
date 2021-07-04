from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats

from lib.helpers import make_paulicliques, estimate_fidelity

import tequila as tq  # noqa


def g(x, y):
    if y < x:
        return 0

    return x * (1.0 - 2.0 * y) + y - 2.0 * np.sqrt(x * (1.0 - x) * y * (1.0 - y))


class RobustnessInterval(ABC):
    def __init__(self, observable, ansatz):
        self._observable = observable
        self._ansatz = ansatz

        self._lower_bound = -np.inf
        self._upper_bound = np.inf
        self._expectation = np.nan

        self._variance = np.nan
        self._fidelity = np.nan

        # confidence intervals for finite sampling
        self._lower_bound_ci = np.nan, np.nan
        self._upper_bound_ci = np.nan, np.nan

    @abstractmethod
    def compute_interval(self, variables, backend, device, noise, samples, fidelity=None, alpha=1e-2, nreps=30):
        pass

    @staticmethod
    def finite_sampling_error(num_terms, num_samples, confidence_level=1e-2):
        """
        returns finite sampling error delta based on Hoeffding such that num_terms independent random variables sampled
            num_samples times all have true expectation within empirical mean ± delta with probabilit at least
            1 - confidence_level.
        """
        return np.sqrt(-np.log(0.5 * (1.0 - (1.0 - confidence_level) ** (1.0 / num_terms))) / (2 * num_samples))

    def _compute_expectation_and_variance(self, variables, samples, backend, device, noise):
        # compute expectation
        objective = tq.ExpectationValue(U=self._ansatz, H=self._observable)
        expectation = tq.simulate(objective, variables=variables, samples=samples, backend=backend,  # noqa
                                  device=device, noise=noise)

        # compute variance
        objective = tq.ExpectationValue(U=self._ansatz, H=(self._observable - expectation) ** 2)
        variance = tq.simulate(objective, variables=variables, samples=samples, backend=backend,  # noqa
                               device=device, noise=noise)

        return expectation.real, max(0, variance.real)

    def _compute_expectation(self, variables, samples, backend, device, noise):
        # compute expectation
        objective = tq.ExpectationValue(U=self._ansatz, H=self._observable)
        expectation = tq.simulate(objective, variables=variables, samples=samples, backend=backend,  # noqa
                                  device=device, noise=noise)

        return expectation.real

    @property
    def observable(self):
        return self._observable

    @property
    def ansatz(self):
        return self._ansatz

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def expectation(self):
        return self._expectation

    @property
    def variance(self):
        return self._variance

    @property
    def fidelity(self):
        return self._fidelity

    @property
    def lower_bound_ci(self):
        return self._lower_bound_ci

    @property
    def upper_bound_ci(self):
        return self._upper_bound_ci


class EigenvalueInterval(RobustnessInterval):

    def compute_interval(self, variables, backend, device, noise, samples, fidelity=None, alpha=1e-2, nreps=20):
        if fidelity is None:
            # compute exact solution
            hamiltonian_matrix = hamiltonian.to_matrix()  # noqa
            eigvals, eigvecs = np.linalg.eigh(hamiltonian_matrix)

            # compute fidelity
            fidelity = estimate_fidelity(eigvals, eigvecs, self._ansatz, variables, backend, device, noise, samples)

        self._fidelity = fidelity

        if samples is None:
            self._expectation, self._variance = self._compute_expectation_and_variance(
                variables, samples, backend, device, noise)

            if fidelity <= 0:
                self._lower_bound, self._upper_bound = -np.inf, np.inf
                return

            # compute lower bound
            self._lower_bound = self._expectation - np.sqrt(self._variance) * np.sqrt((1.0 - fidelity) / fidelity)

            # compute upper bound
            self._upper_bound = self._expectation + np.sqrt(self._variance) * np.sqrt((1.0 - fidelity) / fidelity)

            return

        # finite sampling: estimate nreps times and average, compute confidence interval
        expectations_sampled, variances_sampled = [], []

        for _ in range(nreps):
            e, v = self._compute_expectation_and_variance(variables, samples, backend, device, noise)
            expectations_sampled.append(e)
            variances_sampled.append(v)

        lower_bounds = [e - np.sqrt(v) * np.sqrt((1.0 - fidelity) / fidelity) for e, v in zip(expectations_sampled,
                                                                                              variances_sampled)]
        upper_bounds = [e + np.sqrt(v) * np.sqrt((1.0 - fidelity) / fidelity) for e, v in zip(expectations_sampled,
                                                                                              variances_sampled)]

        # calculate mean
        self._expectation = np.mean(expectations_sampled)
        self._variance = np.mean(variances_sampled)
        self._lower_bound = np.mean(lower_bounds)
        self._upper_bound = np.mean(upper_bounds)

        # calculate sample variance
        lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)
        upper_bound_variance = np.var(upper_bounds, ddof=1, dtype=np.float64)

        # calculate confidence intervals
        self._lower_bound_ci = (
            self._lower_bound - np.sqrt(lower_bound_variance / nreps) * stats.t.ppf(q=1 - alpha / 2.0, df=nreps - 1),
            self._lower_bound + np.sqrt(lower_bound_variance / nreps) * stats.t.ppf(q=1 - alpha / 2.0, df=nreps - 1))

        self._upper_bound_ci = (
            self._upper_bound - np.sqrt(upper_bound_variance / nreps) * stats.t.ppf(q=1 - alpha / 2.0, df=nreps - 1),
            self._upper_bound + np.sqrt(upper_bound_variance / nreps) * stats.t.ppf(q=1 - alpha / 2.0, df=nreps - 1))


class ExpectationInterval(RobustnessInterval):

    def compute_interval(self, variables, backend, device, noise, samples, fidelity=None, use_grouping=True,
                         use_second_moment=False, normalization_constants=None, alpha=1e-2, nreps=30):
        if use_second_moment:
            self._compute_interval_second_moment(variables, backend, device, noise, samples, fidelity, use_grouping,
                                                 normalization_constants, alpha, nreps)

        else:
            self._compute_interval_first_moment(variables, backend, device, noise, samples, fidelity, use_grouping,
                                                normalization_constants, alpha)

    def _compute_interval_first_moment(self, variables, backend, device, noise, samples, fidelity, use_grouping,
                                       normalization_constants, alpha):

        self._fidelity = fidelity

        if fidelity < 0:
            print("fidelity={}; setting to 0 manually.".format(fidelity))
            self._fidelity = 0.0

        if normalization_constants is not None:
            # compute interval directly based on Theorem 1
            upper_const = normalization_constants['upper']
            lower_const = normalization_constants['lower']

            # compute expectation
            objective = tq.ExpectationValue(U=self._ansatz, H=self._observable)
            expectation = tq.simulate(objective, variables=variables, samples=samples, backend=backend,  # noqa
                                      device=device, noise=noise)

            if samples is None:
                expectation_lower_bound = expectation_upper_bound = expectation
            else:
                # finite sampling error
                # TODO: account for finite sampling errors or make multiple reps
                expectation_lower_bound = expectation_upper_bound = expectation

            self._expectation = expectation

            if fidelity < (upper_const - expectation) / (upper_const - lower_const):
                self._lower_bound = lower_const
            else:
                self._lower_bound = lower_const + (
                        np.sqrt(self._fidelity) * np.sqrt(expectation_lower_bound - lower_const)
                        - np.sqrt(1 - self._fidelity) * np.sqrt(upper_const - expectation_lower_bound)) ** 2

            if fidelity < (expectation_upper_bound - lower_const) / (upper_const - lower_const):
                self._upper_bound = upper_const
            else:
                self._upper_bound = lower_const + (
                        np.sqrt(1 - self._fidelity) * np.sqrt(expectation_upper_bound - lower_const)
                        - np.sqrt(self._fidelity) * np.sqrt(upper_const - expectation_upper_bound)) ** 2

            return

        # if no normalization constants are provided, we compute the robustness interval based on the pauli terms
        self._lower_bound = 0
        self._upper_bound = 0
        self._expectation = 0

        if use_grouping:
            paulicliques = make_paulicliques(self._observable)
            pauli_strings = [ps.naked() for ps in paulicliques]
            pauli_coeffs = [ps.coeff.real for ps in paulicliques]
        else:
            pauli_strings = [ps.naked() for ps in self._observable.paulistrings]
            pauli_coeffs = [ps.coeff.real for ps in self._observable.paulistrings]

        for p_str, p_coeff in zip(pauli_strings, pauli_coeffs):
            # eval pauli
            if str(p_str) == 'I' or len(p_str) == 0:
                pauli_expec = 1.0
                lower_bound = upper_bound = 1.0
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
                pauli_expec = tq.simulate(objective, variables=variables, samples=samples, backend=backend,
                                          device=device, noise=noise)

                # pauli bounds
                min_eigval = -1.0
                max_eigval = 1.0

                if hasattr(p_str, "compute_eigenvalues"):
                    ev = p_str.compute_eigenvalues()
                    min_eigval = min(ev)
                    max_eigval = max(ev)

                pauli_expec = np.clip(pauli_expec, min_eigval, max_eigval)

                if samples is None:
                    pauli_lower = pauli_upper = pauli_expec
                else:
                    fs_err = self.finite_sampling_error(num_terms=len(pauli_strings), num_samples=samples,
                                                        confidence_level=alpha)
                    pauli_lower = pauli_expec - fs_err
                    pauli_upper = pauli_expec + fs_err

                pauli_lower_normalized = np.clip((pauli_lower - min_eigval) / (max_eigval - min_eigval), 0, 1)
                pauli_upper_normalized = np.clip((pauli_upper - min_eigval) / (max_eigval - min_eigval), 0, 1)

                # TODO: with noise, we need to recompute the fidelity for each pauli group (U not unitary in this case)
                lower_bound = g(1.0 - pauli_lower_normalized, self._fidelity)
                upper_bound = 1.0 - g(pauli_upper_normalized, self._fidelity)
                lower_bound = (max_eigval - min_eigval) * lower_bound + min_eigval
                upper_bound = (max_eigval - min_eigval) * upper_bound + min_eigval

            # update total expectation
            self._expectation += p_coeff * pauli_expec
            self._lower_bound += p_coeff * lower_bound if p_coeff > 0 else p_coeff * upper_bound
            self._upper_bound += p_coeff * upper_bound if p_coeff > 0 else p_coeff * lower_bound

    def _compute_interval_second_moment(self, variables, backend, device, noise, samples, fidelity, use_grouping,
                                        normalization_constants, alpha, nreps=20):
        self._fidelity = fidelity

        if fidelity < 0:
            print("fidelity={}; setting to 0 manually.".format(fidelity))
            self._fidelity = 0.0

        if normalization_constants is not None:
            # compute interval directly based on Theorem 1
            lower_const = normalization_constants['lower']

            # compute expectation
            objective = tq.ExpectationValue(U=self._ansatz, H=self._observable)
            expectation = tq.simulate(objective, variables=variables, samples=samples, backend=backend,  # noqa
                                      device=device, noise=noise)

            # compute second moment
            objective = tq.ExpectationValue(U=self._ansatz, H=(self._observable - expectation) ** 2)
            variance = tq.simulate(objective, variables=variables, samples=samples, backend=backend,  # noqa
                                   device=device, noise=noise)

            self._variance = max(0.0, variance.real)
            self._expectation = expectation

            if samples is None:
                self._lower_bound = lower_const + (
                    max(0, np.sqrt(fidelity * (expectation - lower_const)
                                   - np.sqrt((1 - fidelity) * self._variance / (expectation - lower_const))))) ** 2

                return

            # finite sampling: estimate nreps time and average, compute confidence interval
            expectations_sampled, variances_sampled = [], []

            for _ in range(nreps):
                e, v = self._compute_expectation_and_variance(variables, samples, backend, device, noise)
                expectations_sampled.append(e)
                variances_sampled.append(v)

            lower_bounds = [lower_const + (
                max(0, np.sqrt(fidelity * (e - lower_const) - np.sqrt((1 - fidelity) * v / (e - lower_const))))) ** 2
                            for e, v in zip(expectations_sampled, variances_sampled)]

            # calculate mean
            self._expectation = np.mean(expectations_sampled)
            self._variance = np.mean(variances_sampled)
            self._lower_bound = np.mean(lower_bounds)

            # calculate sample variance
            lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)

            # calculate confidence intervals
            self._lower_bound_ci = (
                self._lower_bound - np.sqrt(lower_bound_variance / nreps) * stats.t.ppf(q=1 - alpha,
                                                                                         df=nreps - 1),
                self._lower_bound + np.sqrt(lower_bound_variance / nreps) * stats.t.ppf(q=1 - alpha,
                                                                                         df=nreps - 1))

            return

        raise NotImplementedError('second moment interval not implemented without normalization!')

# class RobustnessIntervalOLD:
#     def __init__(self, hamiltonian, ansatz, variables, fidelity: float = None, confidence_level: float = 0.01):
#         self._hamiltonian = hamiltonian
#         self._ansatz = ansatz
#         self._variables = variables
#         self._fidelity = fidelity
#         self._confidence_level = confidence_level
#
#         if self._fidelity < 0:
#             warnings.warn("fidelity={}; setting to 0 manually.".format(fidelity), RuntimeWarning)
#             self._fidelity = 0.0
#
#         self._lower = 0.0
#         self._upper = 0.0
#         self._expectation = 0.0
#
#     def compute_interval(self, backend, device, noise, samples, use_grouping, use_second_moment):
#
#         # if use_second_moment:
#         #     return self._compute_second_order_interval(backend, device, noise, samples)
#
#         return self._compute_first_order_interval(backend, device, noise, samples, use_grouping)
#
#     def _compute_first_order_interval(self, backend, device, noise, samples, use_grouping):
#
#         # extract pauli terms
#         if not use_grouping:
#             pauli_strings = [ps.naked() for ps in self._hamiltonian.paulistrings]
#             pauli_coeffs = [ps.coeff.real for ps in self._hamiltonian.paulistrings]
#         else:
#             paulicliques = make_paulicliques(self._hamiltonian)
#             pauli_strings = [ps.naked() for ps in paulicliques]
#             pauli_coeffs = [ps.coeff.real for ps in paulicliques]
#
#         # finite sampling error (hoeffding)
#         num_paulis = len(pauli_coeffs)
#         if samples is None:
#             fs_err = 0.0
#         else:
#             fs_err = np.sqrt(-np.log(0.5 * (1 - (1 - self._confidence_level) ** (1.0 / num_paulis))) / (2 * samples))
#
#         for p_str, p_coeff in zip(pauli_strings, pauli_coeffs):
#             # eval pauli
#             if str(p_str) == 'I' or len(p_str) == 0:
#                 pauli_expec = 1.0
#             else:
#                 if hasattr(p_str, "H"):
#                     observable = p_str.H
#                 else:
#                     observable = tq.QubitHamiltonian.from_paulistrings([p_str])
#
#                 if hasattr(p_str, "U"):
#                     U = self._ansatz + p_str.U
#                 else:
#                     U = self._ansatz
#
#                 objective = tq.ExpectationValue(U=U, H=observable)
#                 pauli_expec = tq.simulate(objective, variables=self._variables, samples=samples, backend=backend,
#                                           device=device, noise=noise)
#
#             # pauli bounds
#             min_eigenvalue = -1.0
#             max_eigenvalue = 1.0
#
#             if hasattr(p_str, "compute_eigenvalues"):
#                 ev = p_str.compute_eigenvalues()
#                 min_eigenvalue = min(ev)
#                 max_eigenvalue = max(ev)
#
#             pauli_expec = np.clip(pauli_expec, min_eigenvalue, max_eigenvalue)
#
#             lower_bound, upper_bound = self._compute_interval_single(p_str, pauli_expec, fs_err, self._fidelity,
#                                                                      min_eigenvalue, max_eigenvalue)
#
#             # update total expectation
#             self._expectation += p_coeff * pauli_expec
#             self._lower += p_coeff * lower_bound if p_coeff > 0 else p_coeff * upper_bound
#             self._upper += p_coeff * upper_bound if p_coeff > 0 else p_coeff * lower_bound
#
#     @staticmethod
#     def _compute_interval_single(pauli_string, pauli_expecation, fs_err, fidelity, min_eigenvalue, max_eigenvalue):
#         if str(pauli_string) == 'I' or len(pauli_string) == 0:
#             return 1.0, 1.0
#
#         c = max_eigenvalue - min_eigenvalue
#         d = min_eigenvalue
#
#         x_lower = np.clip((pauli_expecation - fs_err - d) / c, 0, 1)
#         x_upper = np.clip((pauli_expecation + fs_err - d) / c, 0, 1)
#
#         lower_bound = g(1.0 - x_lower, fidelity)
#         upper_bound = 1.0 - g(x_upper, fidelity)
#         lower_bound = c * lower_bound + d
#         upper_bound = c * upper_bound + d
#
#         return lower_bound, upper_bound
#
#     @property
#     def hamiltonian(self):
#         return self._hamiltonian
#
#     @property
#     def ansatz(self):
#         return self._ansatz
#
#     @property
#     def variables(self):
#         return self._variables
#
#     @property
#     def fidelity(self):
#         return self._fidelity
#
#     @property
#     def lower(self):
#         return self._lower
#
#     @property
#     def upper(self):
#         return self._upper
#
#     @property
#     def expectation(self):
#         return self._expectation
