"""Simulations for the paper. Run from code/:  python scripts/simulate.py [--quick]

(i)  Per-person regret / log T of UCB and Thompson sampling on the welfare
     signal, against the closed form D c_min (Theorem 3 / Corollary 4).
(ii) The min-max LP, and the forced-exploration planner that targets its
     solution, against the closed form D c* (Theorem 5).

Writes reports/results.json and ../paper/figures/*.pdf.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regvec import (blog_example, compensable_example, random_instance,  # noqa: E402
                    run_base, run_forced, per_person_regret, minmax, ville_certificate)

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "reports")
FIGS = os.path.join(HERE, "..", "..", "paper", "figures")


def summarise(inst, counts, cps, target_c):
    """counts (C, R, K) -> per-person mean and s.e. of R_i(T)/log T, plus targets."""
    reg = per_person_regret(inst, counts) / np.log(cps)[:, None, None]  # (C, R, N)
    R = counts.shape[1]
    sub = inst.suboptimal
    ratio = counts[:, :, sub].mean(1) / (np.log(cps)[:, None] * target_c[None, :])
    return {
        "T": cps.tolist(),
        "mean": reg.mean(1).tolist(),
        "se": (reg.std(1, ddof=1) / np.sqrt(R)).tolist(),
        "target": (inst.D @ target_c).tolist(),
        "count_ratio": ratio.tolist(),  # E[n_a(T)] / (c_a log T), suboptimal arms
    }


def plot(name, res, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (algo, r) in zip(axes, res.items()):
        T = np.array(r["T"])
        m, se, tgt = np.array(r["mean"]), np.array(r["se"]), np.array(r["target"])
        for i in range(m.shape[1]):
            line, = ax.plot(T, m[:, i], marker="o", ms=3, label=f"person {i + 1}")
            ax.fill_between(T, m[:, i] - 2 * se[:, i], m[:, i] + 2 * se[:, i],
                            color=line.get_color(), alpha=0.2)
            ax.axhline(tgt[i], color=line.get_color(), ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("T")
        ax.set_title(algo)
    axes[0].set_ylabel(r"$\mathbb{E}[R_i(T)]/\log T$")
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"{name}.pdf"))
    plt.close(fig)


def main(quick: bool):
    T_max = 100_000 if quick else 1_000_000
    R = 10 if quick else 20
    cps = np.unique(np.logspace(3, np.log10(T_max), 7).astype(np.int64))
    results = {"T_max": T_max, "seeds": R}
    t0 = time.time()

    # (i) asymptotically optimal planners on the two worked examples
    for name, inst in [("blog", blog_example()), ("compensable", compensable_example())]:
        res = {}
        for algo in ["ucb", "ts"]:
            counts = run_base(inst, T_max, cps, R=R, algo=algo, seed=1)
            res[algo] = summarise(inst, counts, cps, inst.c_min)
        results[f"{name}_base"] = res
        plot(f"{name}_base", res, f"{name}: per-person regret / log T (dashed: $D c_{{\\min}}$)")
        print(f"[{time.time() - t0:6.1f}s] {name} base done", flush=True)

    # random instances, shorter horizon
    rng = np.random.default_rng(7)
    T_rand = min(T_max, 200_000)
    cps_r = np.unique(np.logspace(3, np.log10(T_rand), 5).astype(np.int64))
    rand = []
    for k in range(5):
        inst = random_instance(rng)
        counts = run_base(inst, T_rand, cps_r, R=R, algo="ucb", seed=100 + k)
        s = summarise(inst, counts, cps_r, inst.c_min)
        mm = minmax(inst)
        rand.append({
            "mu": inst.mu.tolist(), "sigma": inst.sigma.tolist(),
            "ucb": s,
            "minmax_value": mm.value, "utilitarian_max": float(inst.burden().max()),
            "price": mm.price, "total_cmin": inst.total(),
            "ville_certificate": ville_certificate(inst) is not None,
        })
    results["random"] = rand
    print(f"[{time.time() - t0:6.1f}s] random instances done", flush=True)

    # (ii) min-max fair planner via forced exploration on the compensable example
    inst = compensable_example()
    mm = minmax(inst)
    g = np.zeros(inst.K)
    g[inst.suboptimal] = mm.c - inst.c_min
    res = {}
    for algo in ["ucb", "ts"]:
        counts = run_forced(inst, g, T_max, cps, R=R, algo=algo, seed=2)
        res[algo] = summarise(inst, counts, cps, mm.c)
    # Finite-T check of the construction: forced minus base should be D g log T,
    # whether or not the base planner has reached its asymptote.
    shift = {}
    for algo in ["ucb", "ts"]:
        base_counts = run_base(inst, T_max, cps, R=R, algo=algo, seed=3)
        forced_counts = run_forced(inst, g, T_max, cps, R=R, algo=algo, seed=4)
        d = (per_person_regret(inst, forced_counts).mean(1)
             - per_person_regret(inst, base_counts).mean(1)) / np.log(cps)[:, None]
        shift[algo] = {"T": cps.tolist(), "observed": d.tolist(),
                       "predicted": (inst.delta @ g).tolist()}
    results["compensable_minmax"] = {"c_star": mm.c.tolist(), "value": mm.value,
                                     "price": mm.price, "sim": res, "shift": shift}
    plot("compensable_minmax", res, "compensable: forced exploration to the min-max point (dashed: $D c^*$)")
    print(f"[{time.time() - t0:6.1f}s] forced planner done", flush=True)

    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, "results_quick.json" if quick else "results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", path)

    # compact console report
    for key in ["blog_base", "compensable_base"]:
        for algo, r in results[key].items():
            print(f"\n{key}/{algo}: target {np.round(r['target'], 1)}")
            for T, m, se in zip(r["T"], r["mean"], r["se"]):
                print(f"  T={T:>8d}  mean {np.round(m, 1)}  se {np.round(se, 1)}")
    r = results["compensable_minmax"]
    print(f"\nmin-max: c*={np.round(r['c_star'], 1)} value={r['value']:.1f} price={r['price']:.1f}")
    for algo, s in r["sim"].items():
        print(f"  {algo}: target {np.round(s['target'], 1)}; T={s['T'][-1]} mean {np.round(s['mean'][-1], 1)}"
              f" se {np.round(s['se'][-1], 1)}")
    for algo, s in r["shift"].items():
        print(f"  shift {algo}: predicted D g = {np.round(s['predicted'], 1)}")
        for T, o in zip(s["T"], s["observed"]):
            print(f"     T={T:>8d} observed {np.round(o, 1)}")

    # How often is the utilitarian planner already min-max fair?
    rng = np.random.default_rng(0)
    freq = []
    for N, K in [(3, 3), (5, 4), (10, 5), (20, 8)]:
        free, price, gain = 0, [], []
        for _ in range(400):
            inst = random_instance(rng, N=N, K=K, min_gap=1e-9)
            m = minmax(inst)
            if ville_certificate(inst) is not None:
                free += 1
            else:
                price.append(m.price / inst.total())
                gain.append(1 - m.value / inst.burden().max())
        freq.append({"N": N, "K": K, "frac_free": free / 400,
                     "median_rel_price": float(np.median(price)) if price else None,
                     "median_rel_reduction": float(np.median(gain)) if gain else None})
    results["fairness_frequency"] = freq
    print("\nfairness frequency (400 random instances each, mu ~ N(0,1), sigma ~ U[.5,1.5]):")
    for f in freq:
        print("  ", f)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)

    print("\nrandom instances (UCB, last T): mean / target per person")
    for d in rand:
        print("  ", np.round(d["ucb"]["mean"][-1], 2), "/", np.round(d["ucb"]["target"], 2),
              f"| minmax {d['minmax_value']:.2f} vs util-max {d['utilitarian_max']:.2f}, price {d['price']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
