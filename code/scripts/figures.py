"""Pedagogical figures for the paper. Run from code/:  python scripts/figures.py

Writes to ../paper/figures/:
    fig_example.pdf       anatomy of the worked example (rewards -> burden vector)
    fig_cone.pdf          the achievable cone of burden vectors, and the min-max point
    fig_trajectories.pdf  simulated per-person regret against the predicted limits
                          (reads reports/results.json, written by simulate.py)

Everything except the trajectories is computed from closed forms in regvec.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regvec import (blog_example, compensable_example, status_quo_example, minmax,  # noqa: E402
                    minmax_conflict, ir_optimum)

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "reports")
FIGS = os.path.join(HERE, "..", "..", "paper", "figures")

# One colour per person, fixed across every figure. Persons 1 and 2 are
# identical in both worked examples, so they share a mark and a label.
P12, P3 = "#2a78d6", "#eb6834"
TOTAL = "#6b6a63"       # the collective (neutral)
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "mathtext.fontset": "cm", "pdf.fonttype": 42,
})


def fig_example():
    inst = blog_example()
    names = [r"$a^\star$", r"$b$", r"$c$"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                                 gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Left: what each policy pays each person.
    x = np.arange(3)
    ax.plot(x, inst.mu[0], color=P12, ls="none", marker="o", ms=8, mec="white", mew=1.5, zorder=3)
    for xi in x:
        ax.plot([xi, xi], [inst.mu[0, xi], inst.mu[2, xi]], color=GRID, lw=2.5, zorder=1)
    ax.plot(x, inst.mu[2], color=P3, ls="none", marker="o", ms=8, mec="white", mew=1.5, zorder=4)
    # Where all three people get the same reward, draw one neutral mark.
    same = [xi for xi in x if np.ptp(inst.mu[:, xi]) == 0]
    ax.plot(same, inst.mu[0, same], color=INK, ls="none", marker="o", ms=8, mec="white", mew=1.5, zorder=5)
    ax.text(0.08, 1.12, "everyone", color=INK, va="center")
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(x, [f"{n}\n$W={w:.1f}$" for n, w in zip(names, inst.welfare)])
    ax.set_xlim(-0.35, 2.85)
    ax.set_ylim(-0.4, 1.9)
    ax.set_ylabel("mean reward per round")
    ax.text(1.08, 1.56, "persons 1, 2", color=INK, va="center")
    ax.text(1.08, -0.16, "person 3", color=INK, va="center")
    ax.plot([0, 1], [1.0, 1.0], color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.annotate("", xy=(0.88, -0.1), xytext=(0.88, 1.0),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
    ax.text(0.82, 0.45, "person 3 loses\n$\\delta_{3,b}=1.1$", color=MUTED, ha="right", va="center", fontsize=8)
    ax.text(2.08, 0.5, "everyone", color=INK, va="center")
    ax.set_title("What each policy pays each person")

    # Right: the burden vector, against the collective's total.
    b = inst.burden()
    rows = [("person 1", b[0], P12), ("person 2", b[1], P12),
            ("person 3", b[2], P3), ("collective\n(sum)", inst.total(), TOTAL)]
    y = np.arange(len(rows))[::-1]
    bx.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.55, zorder=3)
    bx.axvline(0, color=MUTED, lw=0.8, zorder=4)
    bx.set_yticks(y, [r[0] for r in rows])
    bx.tick_params(axis="y", length=0, labelcolor=INK)
    for yi, (_, v, _) in zip(y, rows):
        bx.text(max(v, 0) + 14, yi, f"{v:+.0f}", va="center", ha="left")
    bx.set_xlim(-430, 790)
    bx.set_xlabel(r"regret per unit of $\log T$ (negative = gains)")
    bx.spines["left"].set_visible(False)
    bx.xaxis.grid(True, color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    bx.set_title("Who pays for learning that $a^\\star$ is best")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIGS, "fig_example.pdf"))
    plt.close(fig)


def fig_cone():
    inst = compensable_example()
    D, cmin = inst.D, inst.c_min
    apex = D @ cmin                       # (r1, r2, r3); r1 == r2
    mm = minmax(inst)
    A = np.array([apex[0], apex[2]])
    M = np.array([mm.burden[0], mm.burden[2]])
    dirs = {name: np.array([D[0, k], D[2, k]]) for k, name in enumerate("bcd")}

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    xlim, ylim = (-560, 420), (-130, 1130)

    # The cone: everything reachable from the apex by trying arms more often.
    far = 4000.0
    poly = np.array([A + far * dirs["b"], A, A + far * dirs["d"], [far, far]])
    ax.fill(poly[:, 0], poly[:, 1], color=P12, alpha=0.10, lw=0, zorder=1)

    # Equal-burden diagonal and iso-total lines (2 r1 + r3 = const).
    t = np.array(xlim)
    ax.plot(t, t, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(330, 372, "equal burdens", color=MUTED, rotation=37, ha="center", fontsize=8)
    for tot in (inst.total(), inst.total() + mm.price):
        ax.plot(t, tot - 2 * t, color=GRID, lw=1.0, zorder=0)
    ax.text(-457, 1046, "total 94", color=MUTED, fontsize=8, rotation=-62, ha="center", va="center")
    ax.text(-354, 1004, "total 256", color=MUTED, fontsize=8, rotation=-62, ha="center", va="center")

    # Generators: one direction per suboptimal arm.
    lab = {"b": ("try $b$ more", (-12, 4), "right"),
           "c": ("try $c$ more", (8, 2), "left"),
           "d": ("try $d$ more", (12, 6), "left")}
    for name, v in dirs.items():
        tip = A + 190 * v / np.linalg.norm(v)
        ax.annotate("", xy=tip, xytext=A,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=4, shrinkB=0))
        text, off, ha = lab[name]
        ax.annotate(text, xy=tip, xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=8)

    # Path from the utilitarian apex to the min-max point: only arm d is added.
    ax.plot([A[0], M[0]], [A[1], M[1]], color=P3, lw=1.6, zorder=3)
    ax.plot(*A, marker="o", ms=8, color=INK, mec="white", mew=1.5, zorder=5)
    ax.plot(*M, marker="o", ms=8, color=P3, mec="white", mew=1.5, zorder=5)
    ax.annotate("asymptotically optimal planner\n$Dc^{\\min}$: person 3 pays 571",
                xy=A, xytext=(-548, 330), fontsize=8, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=6))
    ax.annotate("min-max point: all pay 85\n(810 extra tries of $d$ per $\\log T$)",
                xy=M, xytext=(125, 20), fontsize=8, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=6))
    ax.text(150, 760, "achievable by a\nconsistent planner", color=P12, fontsize=8.5, ha="center")
    ax.text(-380, 60, "not achievable", color=MUTED, fontsize=8.5, ha="center")

    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"burden of persons 1 and 2 (each), per unit of $\log T$")
    ax.set_ylabel(r"burden of person 3, per unit of $\log T$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_cone.pdf"))
    plt.close(fig)


def fig_trajectories():
    path = os.path.join(REPORTS, "results.json")
    if not os.path.exists(path):
        print("no reports/results.json; run simulate.py first")
        return
    res = json.load(open(path))["blog_base"]
    titles = {"ucb": "UCB on the welfare signal", "ts": "Thompson sampling"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, algo in zip(axes, ("ucb", "ts")):
        r = res[algo]
        T, m, se, tgt = (np.array(r[k]) for k in ("T", "mean", "se", "target"))
        for i, col, name in ((0, P12, "persons 1, 2"), (2, P3, "person 3")):
            ax.axhline(tgt[i], color=col, lw=1.2, ls=(0, (4, 3)), zorder=2)
            ax.fill_between(T, m[:, i] - 2 * se[:, i], m[:, i] + 2 * se[:, i],
                            color=col, alpha=0.15, lw=0)
            ax.plot(T, m[:, i], color=col, lw=1.6, marker="o", ms=3.5)
            if algo == "ucb":
                ax.text(1.15e3, tgt[i] + (28 if i == 2 else -28), f"{name}: limit {tgt[i]:+.0f}",
                        color=INK, fontsize=8, va="bottom" if i == 2 else "top")
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("horizon $T$")
        ax.set_title(titles[algo])
    axes[0].set_ylabel(r"$\mathbb{E}[R_i(T)]\,/\log T$")
    axes[0].set_ylim(-420, 800)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIGS, "fig_trajectories.pdf"))
    plt.close(fig)


def fig_sharing():
    """Utility-possibility set of the status-quo example under time-sharing."""
    inst = status_quo_example()
    mu = inst.mu
    pts = {n: np.array([mu[0, k], mu[2, k]]) for k, n in enumerate(["astar", "b", "c", "a0"])}
    ir = ir_optimum(mu, 3, 0.0).utility
    mm = minmax_conflict(mu).utility
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    hull = np.array([pts["b"], pts["astar"], pts["c"], pts["a0"]])
    ax.fill(hull[:, 0], hull[:, 1], color=P12, alpha=0.10, lw=0, zorder=1)
    ax.plot(*np.vstack([hull, hull[:1]]).T, color=P12, lw=0.8, alpha=0.5, zorder=2)
    # everyone at least as well off as under the status quo
    a0 = pts["a0"]
    ax.plot([a0[0], a0[0]], [a0[1], 1.3], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.plot([a0[0], 1.75], [a0[1], a0[1]], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    # the part of the set where no one is worse off than under a0
    xe = 1.0 + (1.0 - a0[1]) / 2.2            # where the a*-b edge meets person 3's floor
    tri = np.array([a0, [a0[0], ir[2]], [xe, a0[1]]])
    ax.fill(tri[:, 0], tri[:, 1], color=P3, alpha=0.25, lw=0, zorder=2)
    ax.annotate("no one worse off\nthan under $a_0$", xy=(1.25, 0.3), xytext=(1.45, 0.42),
                fontsize=8, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    # welfare contours u3 = W - 2 u1
    for W in (2.6, 3.0):
        x = np.array([0.4, 1.7])
        ax.plot(x, W - 2 * x, color=GRID, lw=1.0, zorder=0)
    ax.text(0.84, 1.26, "welfare 3.0", color=MUTED, fontsize=8, rotation=-56, ha="center")
    ax.text(0.71, 1.12, "welfare 2.6", color=MUTED, fontsize=8, rotation=-56, ha="center")
    lab = {"astar": ("$a^\\star$: utilitarian\noptimum", (-8, 4), "right"),
           "b": ("$b$", (8, 0), "left"), "c": ("$c$", (-8, 0), "right"),
           "a0": ("$a_0$: status quo", (0, -11), "center")}
    for n, q in pts.items():
        ax.plot(*q, marker="o", ms=7, color=INK, mec="white", mew=1.5, zorder=5)
        t, off, ha = lab[n]
        ax.annotate(t, xy=q, xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=8.5)
    ax.plot(ir[0], ir[2], marker="o", ms=8, color=P3, mec="white", mew=1.5, zorder=6)
    ax.annotate("best time-share with no one\nworse off: 60% $a^\\star$, 40% $b$", xy=(ir[0], ir[2]),
                xytext=(1.34, 0.86), fontsize=8, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=5))
    ax.plot(mm[0], mm[2], marker="D", ms=6, color=P12, mec="white", mew=1.5, zorder=6)
    ax.annotate("equal conflict loss:\n69% $a^\\star$, 31% $b$", xy=(mm[0], mm[2]),
                xytext=(1.30, 1.12), fontsize=8, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=5))
    ax.set_xlim(0.4, 1.75)
    ax.set_ylim(-0.2, 1.3)
    ax.set_xlabel("reward per round of persons 1 and 2 (each)")
    ax.set_ylabel("reward per round of person 3")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_sharing.pdf"))
    plt.close(fig)


def _ext():
    path = os.path.join(REPORTS, "extensions.json")
    if not os.path.exists(path):
        print("no reports/extensions.json; run extensions.py first")
        return None
    return json.load(open(path))


def fig_planner():
    res = _ext()
    if res is None:
        return
    r = res["planner"]
    T = np.array(r["T"])
    pc, se, ucb = np.array(r["pc_mean"]), np.array(r["pc_se"]), np.array(r["ucb_mean"])
    keep = T >= 100
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, y, title in ((axes[0], ucb, "Utilitarian learner (UCB)"),
                         (axes[1], pc, "Individually rational planner")):
        for i, col, name in ((0, P12, "persons 1, 2"), (2, P3, "person 3")):
            if y is pc:
                ax.fill_between(T[keep], (y[:, i] - 2 * se[:, i])[keep], (y[:, i] + 2 * se[:, i])[keep],
                                color=col, alpha=0.15, lw=0)
            ax.plot(T[keep], y[keep, i], color=col, lw=1.6)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
        ax.axhline(-r["alpha"], color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("round $t$")
        ax.set_title(title)
    axes[0].text(120, -r["alpha"] - 0.035, "floor: status quo $-\\alpha$", fontsize=8, va="top")
    axes[0].text(2.2e5, ucb[-1, 2], "person 3", fontsize=8, va="center", ha="left", clip_on=False)
    axes[0].text(2.2e5, ucb[-1, 0], "persons\n1, 2", fontsize=8, va="center", ha="left", clip_on=False)
    axes[1].text(2.2e5, pc[-1, 2], "person 3", fontsize=8, va="center", ha="left", clip_on=False)
    axes[1].text(2.2e5, pc[-1, 0], "persons\n1, 2", fontsize=8, va="center", ha="left", clip_on=False)
    axes[0].set_ylabel("average gain per round\nover the status quo")
    fig.tight_layout(w_pad=4.0, rect=(0, 0, 0.94, 1))
    fig.savefig(os.path.join(FIGS, "fig_planner.pdf"))
    plt.close(fig)


def fig_reform():
    res = _ext()
    if res is None:
        return
    r = res["reform"]
    style = {"no friction": (INK, "-"), "switching cost 2": (P3, "-"),
             "terms of 4 rounds": (P12, "-"), "both": (MUTED, (0, (4, 3)))}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    T = np.array(r["T"])
    for name, (col, ls) in style.items():
        ax.plot(T, r["threshold"][name], color=col, ls=ls, lw=1.6, marker="o", ms=3, label=name)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
    ax.set_xlabel("horizon $T$ (rounds)")
    ax.set_ylabel("lowest prior mean at which\nthe reform is tried (prior s.d.)")
    ax.set_title("How bad a reform is worth trying")
    th = np.array(r["theta"])
    for name, (col, ls) in style.items():
        bx.plot(th, r["rounds"][name], color=col, ls=ls, lw=1.6, label=name)
    bx.axvline(0, color=MUTED, lw=0.8, zorder=0)
    bx.set_xlabel("true welfare of the reform (prior s.d.)")
    bx.set_ylabel("expected rounds in force, of 40")
    bx.set_title("How long a reform stays")
    bx.set_ylim(0, 40)
    bx.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIGS, "fig_reform.pdf"))
    plt.close(fig)


def fig_scorecard():
    path = os.path.join(REPORTS, "scorecard.json")
    if not os.path.exists(path):
        print("no reports/scorecard.json; run scorecard.py first")
        return
    res = json.load(open(path))
    T = np.array(res["T"])
    inst = res["institutions"]
    # (key, label, colour, dash, label offset in points for each panel)
    rows = [
        ("oracle: adopt a* at once", "adopt $a^\\star$ at once", INK, (0, (1, 2)), (0, 0)),
        ("utilitarian learner (Thompson)", "utilitarian learner", P12, "-", (-9, 0)),
        ("pilot, then decide", "pilot, then decide", "#eda100", "-", (0, 9)),
        ("equal-conflict time-share (plug-in)", "time-share", "#6250d6", "-", (-10, 0)),
        ("individually rational, alpha = 0.2", "indiv. rational, $\\alpha=0.2$", P3, (0, (4, 2)), (9, -9)),
        ("individually rational, alpha = 0.05", "indiv. rational, $\\alpha=0.05$", P3, "-", (0, 9)),
        ("keep the status quo (or unanimity)", "keep the status quo", MUTED, "-", (-9, 0)),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.6))
    for key, label, col, ls, _ in rows:
        r = inst[key]
        ax.plot(T, r["welfare"], color=col, ls=ls, lw=1.6, label=label)
        bx.plot(T, np.array(r["gain"])[:, 0], color=col, ls=ls, lw=1.6)
    for a in (ax, bx):
        a.set_xscale("log")
        a.set_xlabel("horizon $T$ (rounds)")
    ax.set_ylabel("average welfare per round")
    ax.set_title("Welfare")
    bx.set_ylabel("average gain per round\nover the status quo")
    bx.set_title("Persons 1 and 2")
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=7.5, handlelength=2.6,
               columnspacing=1.4)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(FIGS, "fig_scorecard.pdf"))
    plt.close(fig)


def fig_compare():
    """Per-person floors against compensation with a pooled floor."""
    path = os.path.join(REPORTS, "scorecard.json")
    if not os.path.exists(path):
        return
    res = json.load(open(path))
    if "compensation with a pooled floor, alpha = 0.05" not in res["institutions"]:
        print("scorecard.json has no compensation rows; rerun scorecard.py")
        return
    T, inst = np.array(res["T"]), res["institutions"]
    rows = [
        ("utilitarian learner (Thompson) with compensation", "utilitarian learner, compensated (no floor)", MUTED, (0, (1, 2))),
        ("compensation with a pooled floor, alpha = 0.2", "compensation, pooled floor, $\\alpha=0.2$", P12, (0, (4, 2))),
        ("compensation with a pooled floor, alpha = 0.05", "compensation, pooled floor, $\\alpha=0.05$", P12, "-"),
        ("individually rational, alpha = 0.2", "per-person floors, $\\alpha=0.2$", P3, (0, (4, 2))),
        ("individually rational, alpha = 0.05", "per-person floors, $\\alpha=0.05$", P3, "-"),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.6))
    for key, label, col, ls in rows:
        r = inst[key]
        ax.plot(T, r["welfare"], color=col, ls=ls, lw=1.6, label=label)
        bx.plot(T, r["worst_gain"], color=col, ls=ls, lw=1.6)
    ax.axhline(2.6, color=GRID, lw=0.8, zorder=0)
    bx.axhline(0, color=GRID, lw=0.8, zorder=0)
    for a in (ax, bx):
        a.set_xscale("log")
        a.set_xlabel("horizon $T$ (rounds)")
    ax.set_ylabel("average welfare per round")
    ax.set_title("Welfare")
    bx.set_ylabel("average gain per round\nover the status quo")
    bx.set_title("The person who gains least")
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=7.5, handlelength=2.6,
               columnspacing=1.4)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(FIGS, "fig_compare.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_example()
    fig_cone()
    fig_trajectories()
    fig_sharing()
    fig_planner()
    fig_reform()
    fig_scorecard()
    fig_compare()
    print("wrote fig_example.pdf, fig_cone.pdf, fig_trajectories.pdf to", os.path.abspath(FIGS))
