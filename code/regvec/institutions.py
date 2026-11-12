"""Institutions as learning planners, for the finite-horizon scorecard.

Every planner starts knowing only the status quo a0 (its means are known) and
is scored by the expected reward it gives each person, averaged over rounds.
Functions return `avg[h, r, i]`: person i's average expected reward per round
up to horizon h, in run r.
"""
from __future__ import annotations

import math

import numpy as np

from .instance import Instance
from .safe import run_pc, _lp_times
from .sharing import minmax_conflict


def run_index(inst: Instance, a0: int, horizons, R: int = 200, algo: str = "ucb",
              seed: int = 0) -> np.ndarray:
    """UCB or Thompson sampling on the welfare signal, with W(a0) known."""
    rng = np.random.default_rng(seed)
    W, s, K = inst.welfare, math.sqrt(inst.sigma2), inst.K
    others = [a for a in range(K) if a != a0]
    hs = np.asarray(horizons)
    n, sums = np.zeros((R, K)), np.zeros((R, K))
    rows = np.arange(R)
    out, j = np.empty((len(hs), R, inst.N)), 0
    for t in range(1, int(hs[-1]) + 1):
        if t <= len(others):
            a = np.full(R, others[t - 1])
        else:
            mean = np.where(n > 0, sums / np.maximum(n, 1), 0.0)
            if algo == "ucb":
                idx = mean + np.sqrt(2.0 * inst.sigma2 * math.log(1.0 + t * math.log(t) ** 2)
                                     / np.maximum(n, 1))
            else:
                idx = mean + s / np.sqrt(np.maximum(n, 1)) * rng.standard_normal((R, K))
            idx[:, a0] = W[a0]
            a = np.argmax(idx, axis=1)
        n[rows, a] += 1
        sums[rows, a] += W[a] + s * rng.standard_normal(R)
        if t == hs[j]:
            out[j] = n @ inst.mu.T / t
            j += 1
    return out


def run_pilot(inst: Instance, a0: int, horizons, R: int = 200, seed: int = 0,
              scale: float = 1.0) -> np.ndarray:
    """Pilot, then decide: try every alternative for m rounds, then keep the
    policy with the best observed total (the status quo's total is known).
    m = scale * T^(2/3) / (K - 1), at least 1; the planner knows T."""
    rng = np.random.default_rng(seed)
    W, K = inst.welfare, inst.K
    others = np.array([a for a in range(K) if a != a0])
    out = np.empty((len(horizons), R, inst.N))
    for j, T in enumerate(horizons):
        m = int(min(max(1, round(scale * T ** (2 / 3) / len(others))), T // len(others)))
        est = np.full((R, K), W[a0])
        est[:, others] = W[others] + math.sqrt(inst.sigma2 / m) * rng.standard_normal((R, len(others)))
        pick = np.argmax(est, axis=1)
        n = np.zeros((R, K))
        n[:, others] = m
        n[np.arange(R), pick] += T - m * len(others)
        out[j] = n @ inst.mu.T / T
    return out


def run_share(inst: Instance, a0: int, horizons, R: int = 40, seed: int = 0) -> np.ndarray:
    """Plug-in time-share: play the estimated equal-conflict time-share, mixed
    with uniform exploration of weight t^(-1/3). Observes every person."""
    mu, sig, K, N = inst.mu, inst.sigma, inst.K, inst.N
    others = [a for a in range(K) if a != a0]
    hs = np.asarray(horizons)
    solve_at = set(_lp_times(int(hs[-1])).tolist())
    out = np.empty((len(hs), R, N))
    uni = np.zeros(K)
    uni[others] = 1.0 / len(others)
    for r in range(R):
        rng = np.random.default_rng(seed + r)
        n, sums, tot, j = np.zeros(K), np.zeros((N, K)), np.zeros(N), 0
        q = uni.copy()
        for t in range(1, int(hs[-1]) + 1):
            if t <= len(others):
                p = np.zeros(K)
                p[others[t - 1]] = 1.0
            else:
                if t in solve_at:
                    est = sums / np.maximum(n, 1)
                    est[:, a0] = mu[:, a0]
                    q = minmax_conflict(est).p
                g = t ** (-1.0 / 3.0)
                p = (1 - g) * q + g * uni
            a = rng.choice(K, p=p)
            tot += mu @ p
            n[a] += 1
            sums[:, a] += mu[:, a] + sig * rng.standard_normal(N)
            if t == hs[j]:
                out[j, r] = tot / t
                j += 1
    return out


def run_ir(inst: Instance, a0: int, alpha: float, L: float, horizons, R: int = 40,
           seed: int = 0) -> np.ndarray:
    """The individually rational planner of regvec.safe."""
    hs = np.asarray(horizons)
    out = np.empty((len(hs), R, inst.N))
    for r in range(R):
        rec, U, _, _ = run_pc(inst, a0, alpha, L, int(hs[-1]), seed=seed + r, record=hs)
        out[:, r] = U / rec[:, None] + inst.mu[:, a0]
    return out
