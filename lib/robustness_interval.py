from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
from typing import *

from lib.helpers import PauliClique

EXPECTATION_INTERVAL_TYPES = ['expectation']
EIGENVALUE_INTERVAL_TYPES = ['eigenvalue', 'eigenvalues', 'eigval', 'eigvals']

ROBUSTNESS_INTERVAL_TYPES = EXPECTATION_INTERVAL_TYPES + EIGENVALUE_INTERVAL_TYPES
ROBUSTNESS_INTERVAL_METHODS = ['sdp', 'gramian', 'gram']


class RobustnessInterval(ABC):
    def __init__(self, statistics: Dict[str, Union[List[List[float]], List[float], float]], fidelity: float,
                 confidence_level):
        # clean stats keys
        self._statistics = statistics

        self._fidelity = fidelity
        self._confidence_level = confidence_level

        self._lower_bound = None
        self._upper_bound = None

        self._mean_expectation = None
        self._mean_variance = None

    @abstractmethod
    def _compute_interval(self, *args, **kwargs):
        pass

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def fidelity(self):
        return self._fidelity

    @property
    def expectation(self):
        return self._mean_expectation

    @property
    def variance(self):
        return self._mean_variance


class SDPInterval(RobustnessInterval):

    def __init__(self, statistics: Dict[str, Union[List[List[float]], List[float], float]], fidelity: float,
                 confidence_level: float):
        super(SDPInterval, self).__init__(statistics, fidelity, confidence_level)

        self._pauligroups_coeffs = self._statistics.get('pauliclique_coeffs')
        self._pauligroups_expectations = self._statistics.get('pauliclique_expectations')
        self._pauligroups_eigenvalues = self._statistics.get('pauliclique_eigenvalues')

        if self._pauligroups_coeffs is None:
            self._pauligroups_coeffs = np.ones_like(self._pauligroups_expectations[0]).tolist()

        # pauli strings
        pauli_groups = self._statistics.get('paulicliques', [])
        assert len(pauli_groups) > 0

        self._pauli_groups = []
        for p_group in pauli_groups:
            if isinstance(p_group, PauliClique):
                self._pauli_groups.append(p_group.H.paulistrings)
            else:
                self._pauli_groups.append(p_group)

        # compute mean expectation
        self._mean_expectation = np.mean(np.matmul(self._pauligroups_expectations, self._pauligroups_coeffs))

        self._compute_interval()

    def _compute_interval(self):

        bounds = np.array([self._compute_interval_single(
            self._pauli_groups, self._pauligroups_coeffs, self._pauligroups_eigenvalues, rep_pauli_expectaions)
            for rep_pauli_expectaions in self._pauligroups_expectations])

        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]

        n_reps = len(lower_bounds)

        if n_reps == 1:
            self._lower_bound = lower_bounds[0]
            self._upper_bound = upper_bounds[0]
        else:
            # one sided confidence interval for lower bound
            lower_bound_mean = np.mean(lower_bounds)
            lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)
            self._lower_bound = lower_bound_mean - np.sqrt(lower_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._confidence_level, df=n_reps - 1)

            # one sided confidence interval for upper bound
            upper_bound_mean = np.mean(upper_bounds)
            upper_bound_variance = np.var(upper_bounds, ddof=1, dtype=np.float64)
            self._upper_bound = upper_bound_mean - np.sqrt(upper_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._confidence_level, df=n_reps - 1)

    def _compute_interval_single(self, pauli_strings, pauli_coeffs, pauli_eigvals, pauli_expecs):
        lower_bound = upper_bound = 0.0

        for p_str, p_coeff, p_eigvals, p_expec in zip(pauli_strings, pauli_coeffs, pauli_eigvals, pauli_expecs):

            min_eigval = min(p_eigvals)
            max_eigval = max(p_eigvals)

            if str(p_str) == 'I' or len(p_str) == 0:
                pauli_lower_bound = pauli_upper_bound = 1.0
            else:
                expec_normalized = np.clip(2 * (p_expec - min_eigval) / (max_eigval - min_eigval) - 1, -1, 1,
                                           dtype=np.float64)

                pauli_lower_bound = (max_eigval - min_eigval) / 2.0 * (
                        1 + self._calc_lower_bound(expec_normalized, self.fidelity)) + min_eigval

                pauli_upper_bound = (max_eigval - min_eigval) / 2.0 * (
                        1 + self._calc_upper_bound(expec_normalized, self.fidelity)) + min_eigval

            lower_bound += p_coeff * pauli_lower_bound if p_coeff > 0 else p_coeff * pauli_upper_bound
            upper_bound += p_coeff * pauli_upper_bound if p_coeff > 0 else p_coeff * pauli_lower_bound

        return lower_bound, upper_bound

    @staticmethod
    def _calc_lower_bound(a, f):
        assert -1.0 <= a <= 1.0, 'data not normalized to [-1, 1]'

        if f < 0.5 * (1 - a):
            return -1.0

        return (2 * f - 1) * a - 2 * np.sqrt(f * (1 - f) * (1 - a ** 2))

    @staticmethod
    def _calc_upper_bound(a, f):
        assert -1.0 <= a <= 1.0, 'data not normalized to [-1, 1]'

        if f < 0.5 * (1 + a):
            return 1.0

        return (2.0 * f - 1.0) * a + 2.0 * np.sqrt(f * (1.0 - f) * (1.0 - a ** 2))


class GramianExpectationBound(RobustnessInterval):

    def __init__(self, statistics: Dict[str, Union[List[List[float]], List[float], float]], fidelity: float,
                 confidence_level: float, method: str = 'cliques'):
        super(GramianExpectationBound, self).__init__(statistics, fidelity, confidence_level)

        self._method = method

        # stats for method == normalize_global
        self._hamiltonian_expectations = self._statistics.get('hamiltonian_expectations')
        self._hamiltonian_variances = self._statistics.get('hamiltonian_variances')
        self._normalization_const = self._statistics.get('normalization_const')

        # stats for method == cliques
        self._pauliclique_expectations = self._statistics.get('pauliclique_expectations')
        self._pauliclique_variances = self._statistics.get('pauliclique_variances')
        self._pauliclique_eigenvalues = self._statistics.get('pauliclique_eigenvalues')

        # pauli strings
        pauli_groups = self._statistics.get('paulicliques', [])

        self._pauli_groups = []
        for p_group in pauli_groups:
            if isinstance(p_group, PauliClique):
                self._pauli_groups.append(p_group.H.paulistrings)
            else:
                self._pauli_groups.append(p_group)

        # compute mean expectation
        if method == 'cliques':
            self._mean_expectation = np.mean(np.sum(self._pauliclique_expectations, axis=1))
            self._mean_variances = np.mean(self._pauliclique_expectations, axis=0)
        else:
            self._mean_expectation = np.mean(self._hamiltonian_expectations)
            self._mean_variance = np.mean(self._hamiltonian_expectations)

        # compute lower bound
        self._compute_interval()

    def _compute_interval(self):

        if self._method == 'normalize_global':
            assert self._normalization_const is not None
            bounds = self._compute_bounds_normalize_global()
        elif self._method == 'cliques':
            bounds = self._compute_bounds_cliques()
        else:
            raise ValueError(f'method must be one of "normalize_global", "cliques"; got {self._method}')

        n_reps = len(bounds)

        if n_reps == 1:
            self._lower_bound = bounds[0]
        else:
            # one sided confidence interval for lower bound
            lower_bound_mean = np.mean(bounds)
            lower_bound_variance = np.var(bounds, ddof=1, dtype=np.float64)
            self._lower_bound = lower_bound_mean - np.sqrt(lower_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._confidence_level, df=n_reps - 1)

    def _compute_bounds_normalize_global(self) -> List[float]:

        lower_bounds = []

        for expectation, variance in zip(self._hamiltonian_expectations, self._hamiltonian_variances):
            lower_bound = -self._normalization_const + self._calc_lower_bound(
                a=self._normalization_const + expectation, v=variance, f=self.fidelity)  # noqa
            lower_bounds.append(lower_bound)

        return lower_bounds

    def _compute_bounds_cliques(self):

        lower_bounds = []

        for pc_expectations, pc_variances in zip(self._pauliclique_expectations, self._pauliclique_variances):

            lower_bound = 0.0
            for clique, eigvals, expec, variance in zip(self._pauli_groups, self._pauliclique_eigenvalues,
                                                        pc_expectations, pc_variances):
                min_eigval = min(eigvals)
                expec_pos = np.clip(expec - min_eigval, 0, None, dtype=np.float)
                clique_lower_bound = min_eigval + self._calc_lower_bound(expec_pos, variance, self.fidelity)

                lower_bound += clique_lower_bound

            lower_bounds.append(lower_bound)

        return lower_bounds

    @staticmethod
    def _calc_lower_bound(a, v, f):
        assert a >= 0

        if f / (1 - f) < v / (a ** 2):
            return 0.0

        return f * a + (1 - f) / a * v - 2 * np.sqrt(v * f * (1 - f))


class GramianEigenvalueInterval(RobustnessInterval):

    def __init__(self, statistics: Dict[str, Union[List[List[float]], List[float], float]], fidelity: float,
                 confidence_level: float):
        super(GramianEigenvalueInterval, self).__init__(statistics, fidelity, confidence_level)

        self._expectations = self._statistics.get('hamiltonian_expectations')
        self._variances = self._statistics.get('hamiltonian_variances')

        # compute mean expectation
        self._mean_expectation = np.mean(self._expectations)
        self._mean_variance = np.mean(self._variances)

        self._compute_interval()

    def _compute_interval(self):

        if self.fidelity <= 0:
            self._lower_bound = -np.inf
            self._upper_bound = np.inf
            return

        lower_bounds, upper_bounds = [], []
        for e, v in zip(self._expectations, self._variances):
            err_term = np.sqrt(v) * np.sqrt((1 - self.fidelity) / self.fidelity)
            lower_bounds.append(e - err_term)
            upper_bounds.append(e + err_term)

        n_reps = len(lower_bounds)

        if n_reps == 1:
            self._lower_bound = lower_bounds[0]
            self._upper_bound = upper_bounds[0]
        else:
            # one sided confidence interval for lower bound
            lower_bound_mean = np.mean(lower_bounds)
            lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)
            self._lower_bound = lower_bound_mean - np.sqrt(lower_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._confidence_level, df=n_reps - 1)

            # one sided confidence interval for upper bound
            upper_bound_mean = np.mean(upper_bounds)
            upper_bound_variance = np.var(upper_bounds, ddof=1, dtype=np.float64)
            self._upper_bound = upper_bound_mean - np.sqrt(upper_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._confidence_level, df=n_reps - 1)

# def compute_robustness_interval(method: str,
#                                 kind: str,
#                                 statistics: Dict[str, Union[List[List[float]], List[float], float]],
#                                 fidelity: float,
#                                 normalization_const: float = None,
#                                 confidence_level: float = 1e-2) -> RobustnessInterval:
#     """
#     convenience function for calculation of robustness intervals
#
#     Args:
#         method: str, method used to compute robustness interval
#         kind: str, type of robustness interval
#         statistics: dict, must contain statistics required for the robustness interval computation.
#         fidelity: float, a lower bound to the fidelity with the target state
#         confidence_level: float, defaults to 1e-2; only relevant when statistics were sampled
#         normalization_const: float, required when kind is set to expectation and method is gramian; ensures that A ≥ 0
#
#     Returns:
#         RobustnessInterval
#     """
#     method = method.lower().replace('_', '').replace(' ', '')
#     kind = kind.lower().replace('_', '').replace(' ', '')
#
#     if kind not in ROBUSTNESS_INTERVAL_TYPES:
#         raise ValueError(
#             f'unknown robustness interval type; got {kind}, but kind must be one of {ROBUSTNESS_INTERVAL_TYPES} ')
#
#     if method not in ROBUSTNESS_INTERVAL_METHODS:
#         raise ValueError(f'unknown method; got {method}, but method must be one of {ROBUSTNESS_INTERVAL_METHODS}')
#
#     if kind in EXPECTATION_INTERVAL_TYPES:
#         if method == 'sdp':
#             return SDPInterval(statistics=statistics,
#                                fidelity=fidelity,
#                                confidence_level=confidence_level)
#
#         if method == 'gramian':
#             if normalization_const is None:
#                 raise ValueError(f'normalization constant required when kind={kind} and method={method}')
#
#             return GramianExpectationBound(statistics=statistics,
#                                            fidelity=fidelity,
#                                            confidence_level=confidence_level)
#
#         raise ValueError(f'unknown method {method} for robustness interval with kind={kind}')
#
#     if kind in EIGENVALUE_INTERVAL_TYPES:
#         return GramianEigenvalueInterval(statistics=statistics,
#                                          fidelity=fidelity,
#                                          confidence_level=confidence_level)
