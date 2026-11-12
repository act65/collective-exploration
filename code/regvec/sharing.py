"""Time-sharing, status-quo baselines and individually rational targets.

A time-share is a distribution p over arms; its long-run utility vector is
mu @ p. Everything here is a small linear program over the simplex.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


def conflict(mu: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-person conflict loss max_a mu[i, a] - (mu p)_i, per round."""
    return mu.max(axis=1) - mu @ p


@dataclass
class ShareResult:
    p: np.ndarray        # distribution over arms
    value: float         # objective value
    utility: np.ndarray  # mu @ p
    welfare: float       # sum of utilities per round
    duals: np.ndarray | None = None  # shadow price of each person's floor


def minmax_conflict(mu: np.ndarray) -> ShareResult:
    """min_p max_i [max_a mu_ia - (mu p)_i]; ties broken by welfare."""
    N, K = mu.shape
    best = mu.max(axis=1)
    A = np.hstack([-mu, -np.ones((N, 1))])          # best_i - mu_i p <= t
    Aeq = np.hstack([np.ones((1, K)), [[0.0]]])
    bounds = [(0, None)] * K + [(None, None)]
    r = linprog(np.r_[np.zeros(K), 1.0], A_ub=A, b_ub=-best, A_eq=Aeq, b_eq=[1.0],
                bounds=bounds, method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    t = r.x[-1]
    # second stage: maximise welfare among (near-)minimisers
    A2 = np.vstack([A, np.r_[np.zeros(K), 1.0]])
    b2 = np.r_[-best, t + 1e-9 * max(1.0, abs(t))]
    r2 = linprog(np.r_[-mu.sum(axis=0), 0.0], A_ub=A2, b_ub=b2, A_eq=Aeq, b_eq=[1.0],
                 bounds=bounds, method="highs")
    p = np.clip((r2.x if r2.status == 0 else r.x)[:K], 0, None)
    p /= p.sum()
    u = mu @ p
    return ShareResult(p=p, value=float(t), utility=u, welfare=float(u.sum()))


def ir_optimum(mu: np.ndarray, a0: int, alpha: float | np.ndarray = 0.0) -> ShareResult:
    """Welfare-best time-share that leaves no one more than alpha per round
    below the status quo a0:  max 1' mu p  s.t.  mu p >= mu[:, a0] - alpha.

    duals[i] is the welfare gained per unit of extra tolerance from person i
    (the price of person i's protection); zero when the floor is slack.
    """
    N, K = mu.shape
    floor = mu[:, a0] - np.broadcast_to(np.asarray(alpha, float), (N,))
    r = linprog(-mu.sum(axis=0), A_ub=-mu, b_ub=-floor, A_eq=np.ones((1, K)), b_eq=[1.0],
                bounds=[(0, None)] * K, method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    p = np.clip(r.x, 0, None)
    p /= p.sum()
    u = mu @ p
    return ShareResult(p=p, value=float(u.sum()), utility=u, welfare=float(u.sum()),
                       duals=-np.asarray(r.ineqlin.marginals))


def revealed_weights(mu: np.ndarray, a0: int) -> np.ndarray | None:
    """Welfare weights w (>= 0, sum 1) closest to uniform in max-norm under
    which a0 maximises sum_i w_i mu[i, a]. None if no weights rationalise a0."""
    N, K = mu.shape
    others = [a for a in range(K) if a != a0]
    # variables (w, s): minimise s with |w_i - 1/N| <= s and w'(mu_a - mu_a0) <= 0
    A = [np.r_[mu[:, a] - mu[:, a0], 0.0] for a in others]
    b = [0.0] * len(others)
    for i in range(N):
        e = np.zeros(N + 1)
        e[i], e[-1] = 1.0, -1.0
        A.append(e.copy()); b.append(1.0 / N)
        e[i] = -1.0
        A.append(e.copy()); b.append(-1.0 / N)
    r = linprog(np.r_[np.zeros(N), 1.0], A_ub=np.array(A), b_ub=np.array(b),
                A_eq=np.r_[np.ones(N), 0.0][None, :], b_eq=[1.0],
                bounds=[(0, None)] * N + [(0, None)], method="highs")
    return r.x[:N] if r.status == 0 else None
