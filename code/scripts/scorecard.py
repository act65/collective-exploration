"""Finite-horizon scorecard of institutions on the status-quo example.
Run from code/:  python scripts/scorecard.py   (about 3 minutes)

Writes reports/scorecard.json; scripts/figures.py draws fig_scorecard.pdf.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regvec import status_quo_example  # noqa: E402
from regvec.institutions import run_index, run_pilot, run_share, run_ir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "reports", "scorecard.json")
A0, L = 3, 1.0
HORIZONS = [30, 100, 300, 1000, 3000, 10000, 30000]


def score(inst, avg):
    """avg[h, r, i] -> summary per horizon (means over runs, s.e. of welfare)."""
    base, best = inst.mu[:, A0], inst.mu.max(axis=1)
    m = avg.mean(axis=1)                                  # (H, N)
    gain = m - base
    w = avg.sum(axis=2)                                   # (H, R)
    return {"reward": m.tolist(), "welfare": w.mean(1).tolist(),
            "welfare_se": (w.std(1, ddof=1) / np.sqrt(w.shape[1])).tolist() if w.shape[1] > 1 else [0.0] * len(m),
            "gain": gain.tolist(), "worst_gain": gain.min(1).tolist(),
            "max_conflict": (best - m).max(1).tolist(),
            "transfers_needed": np.clip(-gain, 0, None).sum(1).tolist(),
            # share of runs in which the worst-off person ends below the status quo
            "p_someone_worse": ((avg - base).min(2) < -1e-9).mean(1).tolist()}


def main():
    inst, t0 = status_quo_example(), time.time()
    H = len(HORIZONS)
    const = lambda a: np.tile(inst.mu[:, a], (H, 1, 1))
    runs = {
        "keep the status quo (or unanimity)": const(A0),
        "oracle: adopt a* at once": const(inst.a_star),
        "utilitarian learner (UCB)": run_index(inst, A0, HORIZONS, algo="ucb", seed=1),
        "utilitarian learner (Thompson)": run_index(inst, A0, HORIZONS, algo="ts", seed=2),
        "pilot, then decide": run_pilot(inst, A0, HORIZONS, seed=3),
    }
    print("index planners done", round(time.time() - t0), "s")
    runs["equal-conflict time-share (plug-in)"] = run_share(inst, A0, HORIZONS, seed=10)
    print("time-share done", round(time.time() - t0), "s")
    for alpha in (0.05, 0.2):
        runs[f"individually rational, alpha = {alpha}"] = run_ir(inst, A0, alpha, L, HORIZONS, seed=100)
    res = {"T": HORIZONS, "institutions": {k: score(inst, v) for k, v in runs.items()}}
    json.dump(res, open(OUT, "w"), indent=1)
    for k, v in res["institutions"].items():
        print(f"\n{k}")
        for j, T in enumerate(HORIZONS):
            print(f"  T={T:>6}  W={v['welfare'][j]:.3f}±{v['welfare_se'][j]:.3f}  gain={np.round(v['gain'][j], 3)}"
                  f"  conflict={v['max_conflict'][j]:.3f}  transfers={v['transfers_needed'][j]:.3f}"
                  f"  P(worse)={v['p_someone_worse'][j]:.2f}")
    print("total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
