"""Numbers for the time-sharing, status-quo and finite-horizon sections.
Run from code/:  python scripts/extensions.py   (about 3 minutes)

Writes reports/extensions.json; scripts/figures.py draws from it.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regvec import (blog_example, status_quo_example, run_base, run_pc, minmax_conflict,  # noqa: E402
                    ir_optimum, revealed_weights, conflict, reform)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "reports", "extensions.json")
A0, ALPHA, LBOUND = 3, 0.05, 1.0


def unit(K, a):
    e = np.zeros(K)
    e[a] = 1.0
    return e


def scorecard(inst):
    """Long-run outcome of each institution on the status-quo example."""
    mu, K, base = inst.mu, inst.K, inst.mu[:, A0]
    rows = {
        "status quo forever": unit(K, A0),
        "utilitarian learner": unit(K, inst.a_star),
        "unanimity (alpha = 0)": ir_optimum(mu, A0, 0.0).p,
        f"tolerance alpha = {ALPHA}": ir_optimum(mu, A0, ALPHA).p,
        "min-max conflict time-share": minmax_conflict(mu).p,
    }
    return {k: {"p": p.tolist(), "utility": (mu @ p).tolist(), "welfare": float((mu @ p).sum()),
                "min_gain": float((mu @ p - base).min()),
                "max_conflict": float(conflict(mu, p).max())} for k, p in rows.items()}


def random_stats(rng, N, K, reps=400):
    share_helps, red, price = 0, [], []
    efficient, recovered0, recovered_a = 0, [], []
    for _ in range(reps):
        mu = rng.normal(size=(N, K))
        W = mu.sum(axis=0)
        astar = int(np.argmax(W))
        # time-sharing against conflict
        c0 = conflict(mu, unit(K, astar)).max()
        m = minmax_conflict(mu)
        if m.value < c0 - 1e-7:
            share_helps += 1
            red.append((c0 - m.value) / c0)
            price.append((W[astar] - m.welfare) / N)
        # unanimity against a random status quo other than a*
        a0 = int(rng.choice([a for a in range(K) if a != astar]))
        r0 = ir_optimum(mu, a0, 0.0)
        if r0.welfare <= W[a0] + 1e-7:
            efficient += 1
        recovered0.append((r0.welfare - W[a0]) / (W[astar] - W[a0]))
        recovered_a.append((ir_optimum(mu, a0, 0.1).welfare - W[a0]) / (W[astar] - W[a0]))
    return {"N": N, "K": K, "share_helps": share_helps / reps,
            "median_conflict_reduction": float(np.median(red)),
            "median_price_per_person": float(np.median(price)),
            "status_quo_blocks_everything": efficient / reps,
            "median_gain_recovered_unanimity": float(np.median(recovered0)),
            "median_gain_recovered_alpha_0.1": float(np.median(recovered_a))}


def planner_runs(inst, T=200_000, R=20):
    base = inst.mu[:, A0]
    gains, worst, final = [], [], []
    for seed in range(R):
        rec, U, p, w = run_pc(inst, A0, ALPHA, LBOUND, T, seed=seed)
        gains.append(U / rec[:, None])
        worst.append(w)
        final.append(p)
    gains = np.array(gains)
    counts = run_base(inst, T, rec, R=R, algo="ucb", seed=100)        # (C, R, K)
    ucb = counts @ (inst.mu - base[:, None]).T / rec[:, None, None]   # (C, R, N)
    return {"T": rec.tolist(), "pc_mean": gains.mean(0).tolist(),
            "pc_se": (gains.std(0, ddof=1) / np.sqrt(R)).tolist(),
            "pc_worst_round_margin": float(min(worst)),
            "pc_final_p": np.mean(final, axis=0).tolist(),
            "ucb_mean": ucb.mean(1).tolist(),
            "target": (ir_optimum(inst.mu, A0, ALPHA).utility - base).tolist(), "alpha": ALPHA}


def reform_runs(noise=3.0):
    settings = {"no friction": (0.0, 1), "switching cost 2": (2.0, 1),
                "terms of 4 rounds": (0.0, 4), "both": (2.0, 4)}
    horizons = [4, 8, 12, 16, 20, 28, 40, 60, 80]
    thetas = np.linspace(-1.5, 1.5, 13)
    out = {"noise": noise, "T": horizons, "theta": thetas.tolist(), "threshold": {}, "rounds": {}}
    for name, (c, L) in settings.items():
        out["threshold"][name] = [reform.solve(T, noise, c, L).threshold() for T in horizons]
        plan = reform.solve(40, noise, c, L)
        out["rounds"][name] = [float(plan.simulate(0.0, th, paths=4000).mean()) for th in thetas]
    return out


def main():
    t0 = time.time()
    inst, rng = status_quo_example(), np.random.default_rng(0)
    res = {"scorecard": scorecard(inst)}
    res["status_quo"] = {
        "privilege_regret": conflict(inst.mu, unit(inst.K, A0)).tolist(),
        "revealed_weights": (lambda w: None if w is None else w.tolist())(revealed_weights(inst.mu, A0)),
        "ir0_duals": ir_optimum(inst.mu, A0, 0.0).duals.tolist(),
        "c_min": inst.c_min.tolist(), "burden": inst.burden().tolist(), "total": inst.total()}
    m = minmax_conflict(blog_example().mu)
    res["blog_share"] = {"p": m.p.tolist(), "value": m.value, "welfare": m.welfare}
    res["random"] = [random_stats(rng, N, K) for N, K in ((3, 3), (5, 4), (10, 5), (20, 8))]
    print("LPs done", round(time.time() - t0), "s")
    res["planner"] = planner_runs(inst)
    print("planner done", round(time.time() - t0), "s")
    res["reform"] = reform_runs()
    json.dump(res, open(OUT, "w"), indent=1)
    for k in ("scorecard", "status_quo", "blog_share", "random"):
        print(k, json.dumps(res[k], indent=1))
    print("planner", res["planner"]["pc_final_p"], res["planner"]["pc_worst_round_margin"],
          res["planner"]["pc_mean"][-1], res["planner"]["ucb_mean"][-1])
    print("reform", json.dumps(res["reform"]["threshold"]), json.dumps(res["reform"]["rounds"]))
    print("total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
