"""Public-arm Gaussian bandit instances and the closed forms from the paper.

Notation follows paper/main.tex:
    mu[i, a]     mean reward of person i under policy (arm) a, shape (N, K)
    sigma[i]     idiosyncratic noise standard deviation of person i, shape (N,)
    tau          standard deviation of a shock common to everyone (default 0)
    a_star       the unique welfare-optimal arm
    delta[i, a]  mu[i, a_star] - mu[i, a]   (individual gap; any sign)
    Delta[a]     sum_i delta[i, a]          (welfare gap; > 0 for a != a_star)
    c_min[a]     2 sigma^2 / Delta[a]^2     (Lai-Robbins rate, sigma^2 = 1' Sigma 1,
                 the variance of the welfare signal)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Instance:
    mu: np.ndarray      # (N, K)
    sigma: np.ndarray   # (N,)
    tau: float = 0.0    # common-shock standard deviation

    def __post_init__(self):
        mu = np.asarray(self.mu, dtype=float)
        sigma = np.asarray(self.sigma, dtype=float)
        if mu.ndim != 2:
            raise ValueError("mu must be (N, K)")
        if sigma.shape != (mu.shape[0],) or np.any(sigma <= 0):
            raise ValueError("sigma must be (N,) and positive")
        if self.tau < 0:
            raise ValueError("tau must be non-negative")
        W = mu.sum(axis=0)
        order = np.sort(W)
        if order[-1] - order[-2] <= 0:
            raise ValueError("welfare-optimal arm must be unique")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)

    # --- basic quantities -------------------------------------------------
    @property
    def N(self) -> int:
        return self.mu.shape[0]

    @property
    def K(self) -> int:
        return self.mu.shape[1]

    @property
    def welfare(self) -> np.ndarray:
        return self.mu.sum(axis=0)

    @property
    def a_star(self) -> int:
        return int(np.argmax(self.welfare))

    @property
    def sigma2(self) -> float:
        """Variance of the welfare signal, 1' Sigma 1 = sum_i sigma_i^2 + N^2 tau^2."""
        return float(np.sum(self.sigma ** 2) + (self.N * self.tau) ** 2)

    @property
    def Sigma(self) -> np.ndarray:
        """Noise covariance, diag(sigma^2) + tau^2 11'."""
        return np.diag(self.sigma ** 2) + self.tau ** 2 * np.ones((self.N, self.N))

    @property
    def suboptimal(self) -> np.ndarray:
        return np.array([a for a in range(self.K) if a != self.a_star])

    @property
    def delta(self) -> np.ndarray:
        """Individual gaps, (N, K); column a_star is zero."""
        return self.mu[:, [self.a_star]] - self.mu

    @property
    def Delta(self) -> np.ndarray:
        """Welfare gaps, (K,); entry a_star is zero."""
        return self.delta.sum(axis=0)

    @property
    def D(self) -> np.ndarray:
        """Individual-gap matrix restricted to suboptimal arms, (N, K-1)."""
        return self.delta[:, self.suboptimal]

    @property
    def kappa(self) -> np.ndarray:
        """Shares kappa[i, a] = delta[i, a] / Delta[a] for suboptimal a, (N, K-1)."""
        return self.D / self.Delta[self.suboptimal]

    # --- closed forms -----------------------------------------------------
    @property
    def c_min(self) -> np.ndarray:
        """Lai-Robbins constants 2 sigma^2 / Delta_a^2 for suboptimal arms, (K-1,)."""
        return 2.0 * self.sigma2 / self.Delta[self.suboptimal] ** 2

    def burden(self, c: np.ndarray | None = None) -> np.ndarray:
        """Per-person asymptotic burden D c (units of log T); default c = c_min."""
        c = self.c_min if c is None else np.asarray(c, dtype=float)
        return self.D @ c

    def total(self, c: np.ndarray | None = None) -> float:
        c = self.c_min if c is None else np.asarray(c, dtype=float)
        return float(self.Delta[self.suboptimal] @ c)

    def kl_alternative(self, d: np.ndarray) -> float:
        """KL between N(m, Sigma) and N(m + d, Sigma)."""
        d = np.asarray(d, dtype=float)
        return float(0.5 * d @ np.linalg.solve(self.Sigma, d))


def blog_example(sigma: float = 1.0) -> Instance:
    """Three people, three policies (a*, polarising b, uniformly worse c)."""
    mu = np.array([
        [1.0, 1.5, 0.5],
        [1.0, 1.5, 0.5],
        [1.0, -0.1, 0.5],
    ])
    return Instance(mu=mu, sigma=np.full(3, sigma))


def compensable_example(sigma: float = 1.0) -> Instance:
    """The blog example plus an arm d that person 3 prefers to a*."""
    mu = np.array([
        [1.0, 1.5, 0.5, 0.6],
        [1.0, 1.5, 0.5, 0.6],
        [1.0, -0.1, 0.5, 1.6],
    ])
    return Instance(mu=mu, sigma=np.full(3, sigma))


def status_quo_example(sigma: float = 1.0) -> Instance:
    """The blog example plus a status quo a0 (last column) that favours
    persons 1 and 2: columns are (a*, b, c, a0)."""
    mu = np.array([
        [1.0, 1.5, 0.5, 1.2],
        [1.0, 1.5, 0.5, 1.2],
        [1.0, -0.1, 0.5, 0.2],
    ])
    return Instance(mu=mu, sigma=np.full(3, sigma))


def random_instance(rng: np.random.Generator, N: int = 5, K: int = 4,
                    min_gap: float = 0.3) -> Instance:
    """Random instance with welfare gaps at least `min_gap` (kept moderate so
    that finite-T simulations are informative)."""
    while True:
        mu = rng.normal(size=(N, K))
        sigma = rng.uniform(0.5, 1.5, size=N)
        W = mu.sum(axis=0)
        s = np.sort(W)
        if np.all(s[-1] - s[:-1] >= min_gap):
            return Instance(mu=mu, sigma=sigma)
