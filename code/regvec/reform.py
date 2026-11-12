"""One known status quo, one unknown reform: the Bayes-optimal planner at a
finite horizon, with switching costs and terms of office.

Units. The status quo has welfare 0 per round. The reform has welfare theta
per round, with prior theta ~ N(m, 1). While the reform is in force the planner
sees theta + noise * N(0, 1) each round. Policies are held for blocks of
`term` rounds; changing policy between blocks costs `switch`. The planner
maximises expected total welfare over `T` rounds, starting at the status quo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_Z, _W = np.polynomial.hermite_e.hermegauss(41)
_W = _W / _W.sum()


@dataclass
class ReformPlan:
    T: int
    noise: float
    switch: float
    term: int
    grid: np.ndarray      # posterior-mean grid
    try_reform: np.ndarray  # [epoch, pulls/term, grid, current policy] -> bool
    value0: np.ndarray    # value at the start as a function of the prior mean

    def threshold(self) -> float:
        """Smallest prior mean at which the planner starts the reform."""
        on = self.try_reform[0, 0, :, 0]
        return float(self.grid[np.argmax(on)]) if on.any() else float("inf")

    def simulate(self, m: float, theta: float, paths: int = 4000, seed: int = 0) -> np.ndarray:
        """Rounds the reform is in force, per path, when the truth is theta."""
        rng = np.random.default_rng(seed)
        E = self.T // self.term
        n = np.zeros(paths, dtype=int)       # blocks of reform so far
        S = np.zeros(paths)                  # sum of observations
        cur = np.zeros(paths, dtype=int)
        rounds = np.zeros(paths)
        for k in range(E):
            pulls = n * self.term
            x = (m + S / self.noise ** 2) / (1.0 + pulls / self.noise ** 2)
            j = np.clip(np.searchsorted(self.grid, x), 0, len(self.grid) - 1)
            go = self.try_reform[k, n, j, cur]
            S = S + go * (self.term * theta + self.noise * np.sqrt(self.term)
                          * rng.standard_normal(paths))
            n = n + go
            rounds += go * self.term
            cur = go.astype(int)
        return rounds


def solve(T: int, noise: float, switch: float = 0.0, term: int = 1,
          grid: np.ndarray | None = None) -> ReformPlan:
    """Backward induction over (epoch, pulls so far, posterior mean, current policy)."""
    if T % term:
        raise ValueError("T must be a multiple of term")
    E = T // term
    grid = np.linspace(-8.0, 8.0, 1601) if grid is None else grid
    var = lambda pulls: 1.0 / (1.0 + pulls / noise ** 2)  # posterior variance
    V = np.zeros((E + 1, len(grid), 2))                   # V[n] at the next epoch
    act = np.zeros((E, E, len(grid), 2), dtype=bool)
    for k in range(E - 1, -1, -1):
        Vn = np.zeros_like(V)
        for n in range(k + 1):
            step = np.sqrt(max(var(n * term) - var((n + 1) * term), 0.0))
            cont = sum(w * np.interp(grid + step * z, grid, V[n + 1, :, 1])
                       for z, w in zip(_Z, _W))
            ref = term * grid + cont
            for cur in (0, 1):
                stay = V[n, :, 0] - (switch if cur == 1 else 0.0)
                go = ref - (switch if cur == 0 else 0.0)
                act[k, n, :, cur] = go > stay
                Vn[n, :, cur] = np.maximum(go, stay)
        V = Vn
    return ReformPlan(T=T, noise=noise, switch=switch, term=term, grid=grid,
                      try_reform=act, value0=V[0, :, 0])
