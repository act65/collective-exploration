"""Linear programs over the achievable cone {D c : c >= c_min}."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .instance import Instance


@dataclass
class MinMaxResult:
    c: np.ndarray            # optimal exploration rates (suboptimal arms)
    value: float             # min over the cone of max_i burden_i
    burden: np.ndarray       # D c
    price: float             # extra utilitarian regret, sum_a Delta_a (c_a - c_min_a)


def minmax(inst: Instance) -> MinMaxResult:
    """min_{c >= c_min} max_i (D c)_i, as an LP in (c, t)."""
    D, cmin = inst.D, inst.c_min
    N, m = D.shape
    obj = np.zeros(m + 1)
    obj[-1] = 1.0
    A = np.hstack([D, -np.ones((N, 1))])
    bounds = [(lo, None) for lo in cmin] + [(None, None)]
    res = linprog(obj, A_ub=A, b_ub=np.zeros(N), bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    c = res.x[:m]
    return MinMaxResult(c=c, value=float(res.x[-1]), burden=D @ c,
                        price=float(inst.Delta[inst.suboptimal] @ (c - cmin)))


def budget_feasible(inst: Instance, beta: np.ndarray) -> bool:
    """Is there c >= c_min with D c <= beta (componentwise)?"""
    D, cmin = inst.D, inst.c_min
    res = linprog(np.zeros(D.shape[1]), A_ub=D, b_ub=np.asarray(beta, float),
                  bounds=[(lo, None) for lo in cmin], method="highs")
    return res.status == 0


def ville_certificate(inst: Instance, tol: float = 1e-9) -> np.ndarray | None:
    """Certificate that c_min is already min-max optimal.

    Let I be the set of persons attaining max_i (D c_min)_i. Returns weights
    y >= 0 on I with sum 1 and y^T D_I >= 0 if they exist (then no direction
    e >= 0 lowers every worst-off burden), else None.
    """
    b = inst.burden()
    I = np.flatnonzero(b >= b.max() - tol * max(1.0, abs(b.max())))
    DI = inst.D[I]
    k = len(I)
    res = linprog(np.zeros(k), A_ub=-DI.T, b_ub=np.zeros(DI.shape[1]),
                  A_eq=np.ones((1, k)), b_eq=[1.0],
                  bounds=[(0, None)] * k, method="highs")
    if res.status != 0:
        return None
    y = np.zeros(inst.N)
    y[I] = res.x
    return y
