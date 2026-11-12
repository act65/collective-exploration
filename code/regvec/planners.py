"""Planners for the public-arm Gaussian bandit, vectorised over independent seeds.

Every planner here observes only the welfare signal Y_t = sum_i X_{t,i}
~ N(W(a_t), sigma^2). The paper shows this loses nothing asymptotically.

Each `run_*` function returns counts n_a(T) at the requested checkpoints,
shape (len(checkpoints), R, K), where R is the number of seeds.
"""
from __future__ import annotations

import math

import numpy as np

from .instance import Instance


def _checkpoints(T_max: int, checkpoints) -> np.ndarray:
    cps = np.unique(np.asarray(checkpoints, dtype=np.int64))
    if cps[-1] > T_max or cps[0] < 1:
        raise ValueError("checkpoints must lie in [1, T_max]")
    return cps


class _SumBandit:
    """State of R parallel copies of a base planner on the welfare signal."""

    def __init__(self, inst: Instance, R: int, rng: np.random.Generator, algo: str):
        self.W = inst.welfare
        self.s = math.sqrt(inst.sigma2)
        self.s2 = inst.sigma2
        self.K = inst.K
        self.R = R
        self.rng = rng
        self.algo = algo
        self.n = np.zeros((R, self.K))
        self.sums = np.zeros((R, self.K))
        self.clock = 0  # the base planner's own time index
        self.rows = np.arange(R)

    def step(self) -> np.ndarray:
        """One step of the base planner; returns the chosen arm per seed."""
        self.clock += 1
        t = self.clock
        if t <= self.K:  # initialise: each arm once
            a = np.full(self.R, t - 1)
        else:
            mean = self.sums / self.n
            if self.algo == "ucb":
                # Asymptotically optimal UCB (Lattimore & Szepesvari 2020, Ch. 8)
                f = 1.0 + t * math.log(t) ** 2
                idx = mean + np.sqrt(2.0 * self.s2 * math.log(f) / self.n)
            elif self.algo == "ts":
                # Gaussian Thompson sampling, known variance, flat prior
                idx = mean + self.s / np.sqrt(self.n) * self.rng.standard_normal((self.R, self.K))
            else:
                raise ValueError(self.algo)
            a = np.argmax(idx, axis=1)
        y = self.W[a] + self.s * self.rng.standard_normal(self.R)
        self.n[self.rows, a] += 1
        self.sums[self.rows, a] += y
        return a


def run_base(inst: Instance, T_max: int, checkpoints, R: int = 20,
             algo: str = "ucb", seed: int = 0) -> np.ndarray:
    """Run UCB or Thompson sampling on the welfare signal."""
    rng = np.random.default_rng(seed)
    cps = _checkpoints(T_max, checkpoints)
    b = _SumBandit(inst, R, rng, algo)
    out = np.empty((len(cps), R, inst.K))
    j = 0
    for t in range(1, T_max + 1):
        b.step()
        if t == cps[j]:
            out[j] = b.n
            j += 1
            if j == len(cps):
                break
    return out


def forced_schedule(g: np.ndarray, T_max: int) -> np.ndarray:
    """Deterministic forced-exploration schedule used in the paper's proof.

    At round t, if some arm b has f_b(t-1) < g_b log t, the round is forced and
    the arm with the largest deficit g_b log t - f_b(t-1) is pulled; otherwise
    the round is handed to the base planner. Returns forced_arm[t-1] in
    {-1 (base step), 0..K-1}.
    """
    g = np.asarray(g, dtype=float)
    f = np.zeros_like(g)
    out = np.full(T_max, -1, dtype=np.int64)
    for t in range(1, T_max + 1):
        deficit = g * math.log(t) - f
        b = int(np.argmax(deficit))
        if deficit[b] > 0:
            out[t - 1] = b
            f[b] += 1
    return out


def run_forced(inst: Instance, g: np.ndarray, T_max: int, checkpoints,
               R: int = 20, algo: str = "ucb", seed: int = 0) -> np.ndarray:
    """Base planner plus deterministic forced exploration at rates g (length K).

    The base planner sees only its own samples, as in the paper's construction.
    """
    rng = np.random.default_rng(seed)
    cps = _checkpoints(T_max, checkpoints)
    sched = forced_schedule(g, T_max)
    b = _SumBandit(inst, R, rng, algo)
    forced = np.zeros(inst.K)
    out = np.empty((len(cps), R, inst.K))
    j = 0
    for t in range(1, T_max + 1):
        if sched[t - 1] >= 0:
            forced[sched[t - 1]] += 1
        else:
            b.step()
        if t == cps[j]:
            out[j] = b.n + forced[None, :]
            j += 1
            if j == len(cps):
                break
    return out


def per_person_regret(inst: Instance, counts: np.ndarray) -> np.ndarray:
    """Pseudo-regret per person, sum_a n_a delta[i, a]; counts (..., K) -> (..., N)."""
    return counts @ inst.delta.T
