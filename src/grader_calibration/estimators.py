"""Grader error-rate estimators.

Each grading context has a symmetric flip rate ``e``. Two priors are used: the matched prior, the
generating distribution ``e = 0.02 + 0.36 U`` with ``U ~ Beta(1.5, 3)``, and a weak historical prior
``Beta(0.84, 5.16)``. Posterior means under the matched prior are computed by Gauss-Jacobi quadrature;
under an ordinary Beta prior they are analytical.

``CountEstimator`` accumulates (possibly weighted) flip and exposure counts from audited answers.
``LikelihoodEstimator`` uses every observed grade: audited answers contribute their flip likelihood and
unaudited ones the likelihood of the grade given the tutor's forecast.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np

COUNT_ORDER = 128
LIKELIHOOD_ORDER = 256
PRIORS = {
    "matched": ("affine_beta", 0.02, 0.36, 1.5, 3.0),
    "historical_weak": ("beta", 0.0, 1.0, 0.84, 5.16),
}


@cache
def quadrature(prior, order):
    """Nodes, log weights, log e and log(1 - e) for the prior, by the Golub-Welsch algorithm."""
    _, lo, scale, a, b = PRIORS[prior]
    # Jacobi exponents of (1 - x) and (1 + x) on [-1, 1].
    alpha, beta = b - 1.0, a - 1.0
    s = alpha + beta
    diagonal = np.empty(order)
    diagonal[0] = (beta - alpha) / (s + 2.0)
    k = np.arange(1, order, dtype=np.float64)
    diagonal[1:] = (beta * beta - alpha * alpha) / ((2 * k + s) * (2 * k + s + 2))
    off = np.empty(order - 1)
    off[0] = math.sqrt(4 * a * b / ((a + b) ** 2 * (a + b + 1)))
    k = np.arange(2, order, dtype=np.float64)
    off[1:] = np.sqrt(4 * k * (k + alpha) * (k + beta) * (k + s) / ((2 * k + s) ** 2 * ((2 * k + s) ** 2 - 1)))
    x, vectors = np.linalg.eigh(np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1))
    weights = vectors[0] ** 2
    positive = weights > 0
    nodes = lo + scale * (x[positive] + 1.0) / 2.0
    weights = weights[positive]
    weights /= np.sum(weights)
    return nodes, np.log(weights), np.log(nodes), np.log1p(-nodes)


def prior_mean(prior):
    _, lo, scale, a, b = PRIORS[prior]
    return lo + scale * a / (a + b)


def _posterior_mean(log_mass, nodes):
    mass = np.exp(log_mass - np.max(log_mass, axis=-1, keepdims=True))
    return np.sum(mass * nodes, axis=-1) / np.sum(mass, axis=-1)


class CountEstimator:
    """Posterior mean from flip count F and exposure N per context."""

    def __init__(self, prior, contexts):
        self.family, _, _, self.a, self.b = PRIORS[prior]
        self.F = np.zeros(contexts)
        self.N = np.zeros(contexts)
        self.means = np.full(contexts, prior_mean(prior))
        if self.family != "beta":
            self.nodes, self.log_weights, self.log_e, self.log_not_e = quadrature(prior, COUNT_ORDER)

    def update(self, context, error, weight=1.0):
        self.F[context] += float(weight) * int(error)
        self.N[context] += float(weight)
        f, n = self.F[context], self.N[context]
        if self.family == "beta":
            self.means[context] = (self.a + f) / (self.a + self.b + n)
        else:
            self.means[context] = _posterior_mean(
                self.log_weights + f * self.log_e + (n - f) * self.log_not_e, self.nodes
            )


class LikelihoodEstimator:
    """Observed-data working likelihood with the tutor's pre-grade forecasts held fixed.

    An unaudited grade ``z`` with forecast ``p`` has probability ``p + (1 - 2p) e`` of being correct;
    the one audited answer per round contributes ``e`` or ``1 - e`` instead.
    """

    def __init__(self, prior, contexts):
        self.nodes, self.log_weights, self.log_e, self.log_not_e = quadrature(prior, LIKELIHOOD_ORDER)
        self.log_likelihood = np.zeros((contexts, len(self.nodes)))
        self._means = np.full(contexts, prior_mean(prior))
        self._stale = np.zeros(contexts, bool)

    @property
    def means(self):
        if np.any(self._stale):
            self._means[self._stale] = _posterior_mean(self.log_weights + self.log_likelihood[self._stale], self.nodes)
            self._stale[:] = False
        return self._means

    def update_round(self, contexts, z, p, selected, verified):
        g = p[:, None] + (1 - 2 * p[:, None]) * self.nodes
        increments = np.where(z[:, None] == 1, np.log(g), np.log1p(-g))
        increments[selected] = self.log_e if verified != z[selected] else self.log_not_e
        np.add.at(self.log_likelihood, contexts, increments)
        self._stale[contexts] = True
