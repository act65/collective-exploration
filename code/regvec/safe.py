"""An individually rational ("Pareto-conservative") planner for public arms.

The status quo a0 has known means. Each round the planner picks a distribution
p_t over arms and must keep every person's expected reward within `alpha` of
the status quo:  (mu p_t)_i >= mu[i, a0] - alpha  for all i and t.

It does so by using lower confidence bounds in the constraints (pessimism) and
upper confidence bounds on welfare in the objective (optimism), and by mixing
in a safe exploration distribution with weight t^(-1/3). The planner knows a
bound `L` with |mu[i, a] - mu[i, a0]| <= L, which is what lets it try an arm
it has never seen. It observes every person's reward.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .instance import Instance


def _lp_times(T: int) -> np.ndarray:
    """Rounds at which the LPs are re-solved: every round early, then ~2% apart."""
    ts, t = [], 1
    while t <= T:
        ts.append(t)
        t = max(t + 1, int(t * 1.02))
    return np.array(ts)


def run_pc(inst: Instance, a0: int, alpha: float, L: float, T: int,
           delta: float = 0.05, seed: int = 0, record=None):
    """Run the planner once. Returns (times, U) where U[j, i] is person i's
    cumulative expected reward minus the status quo's, sum_s (mu p_s)_i - t mu_i0,
    at times[j], the final distribution p_T, and the smallest per-round margin
    min_{t,i} (mu p_t)_i - mu_i0 (which must stay above -alpha). `record`
    defaults to ~200 log-spaced times."""
    rng = np.random.default_rng(seed)
    mu, sig = inst.mu, inst.sigma
    N, K = mu.shape
    others = np.array([a for a in range(K) if a != a0])
    base = mu[:, a0]
    n = np.zeros(K)
    sums = np.zeros((N, K))
    record = np.unique(np.logspace(0, np.log10(T), 200).astype(int)) if record is None else record
    solve_at = set(_lp_times(T).tolist())
    out, gain, j = np.zeros((len(record), N)), np.zeros(N), 0
    p = np.zeros(K)
    p[a0] = 1.0
    worst = 0.0
    s_w = np.sqrt(inst.sigma2)
    ones = np.ones((1, K))
    bnd = [(0, None)] * K
    for t in range(1, T + 1):
        if t in solve_at:
            lcb = np.tile(base[:, None] - L, (1, K))
            ucb_w = np.full(K, base.sum() + N * L)
            lcb[:, a0], ucb_w[a0] = base, base.sum()
            seen = others[n[others] > 0]
            if len(seen):
                m = n[seen]
                # anytime radius: union bound over people, arms and sample sizes
                rad = np.sqrt(2.0 * np.log(4.0 * N * K * m * (m + 1) / delta) / m)
                mean = sums[:, seen] / m
                lcb[:, seen] = np.clip(mean - sig[:, None] * rad, base[:, None] - L, base[:, None] + L)
                ucb_w[seen] = np.minimum(mean.sum(axis=0) + s_w * rad, base.sum() + N * L)
            b_ub = -(base - alpha)
            # exploit: optimistic welfare, pessimistic floors
            r = linprog(-ucb_w, A_ub=-lcb, b_ub=b_ub, A_eq=ones, b_eq=[1.0], bounds=bnd, method="highs")
            q = np.clip(r.x, 0, None)
            # explore: largest equal mass on every non-status-quo arm that stays safe
            A = np.hstack([-lcb, np.zeros((N, 1))])
            for a in others:
                row = np.zeros(K + 1)
                row[a], row[-1] = -1.0, 1.0
                A = np.vstack([A, row])
            r2 = linprog(np.r_[np.zeros(K), -1.0], A_ub=A, b_ub=np.r_[b_ub, np.zeros(len(others))],
                         A_eq=np.hstack([ones, [[0.0]]]), b_eq=[1.0], bounds=bnd + [(0, None)],
                         method="highs")
            e = np.clip(r2.x[:K], 0, None)
            g = min(1.0, t ** (-1.0 / 3.0))
            p = (1 - g) * q / q.sum() + g * e / e.sum()
        a = rng.choice(K, p=p)
        step = mu @ p - base
        worst = min(worst, float(step.min()))
        gain += step
        n[a] += 1
        sums[:, a] += mu[:, a] + sig * rng.standard_normal(N)
        if j < len(record) and t == record[j]:
            out[j] = gain
            j += 1
    return record, out, p, worst
