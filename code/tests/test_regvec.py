"""Tests compare simulations or independent computations with the paper's
closed forms; none checks a closed form against itself."""
import math
import os
import sys

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regvec import (Instance, blog_example, compensable_example, random_instance,  # noqa: E402
                    run_base, run_forced, forced_schedule, per_person_regret,
                    minmax, budget_feasible, ville_certificate, status_quo_example,
                    minmax_conflict, ir_optimum, revealed_weights, conflict, run_pc, reform)


# --- Lemma 2: the KL minimisation --------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_kl_minimisation_numerical(seed):
    """Numerically minimise sum d_i^2/(2 sigma_i^2) s.t. sum d_i = Delta and
    compare with the closed form Delta^2/(2 sigma^2), minimiser d_i ∝ sigma_i^2."""
    rng = np.random.default_rng(seed)
    N = 6
    sig = rng.uniform(0.3, 2.0, N)
    Delta = rng.uniform(0.1, 2.0)
    obj = lambda d: np.sum(d ** 2 / (2 * sig ** 2))
    cons = {"type": "eq", "fun": lambda d: np.sum(d) - Delta}
    res = minimize(obj, x0=np.full(N, Delta / N), constraints=[cons], method="SLSQP",
                   options={"ftol": 1e-14, "maxiter": 500})
    s2 = np.sum(sig ** 2)
    assert res.fun == pytest.approx(Delta ** 2 / (2 * s2), rel=1e-6)
    np.testing.assert_allclose(res.x, Delta * sig ** 2 / s2, rtol=1e-4, atol=1e-7)


def test_kl_alternative_matches_gaussian_kl_by_sampling():
    """Monte Carlo KL between the product Gaussians agrees with kl_alternative."""
    rng = np.random.default_rng(0)
    inst = Instance(mu=np.array([[1.0, 0.0], [0.0, 0.5]]), sigma=np.array([0.7, 1.3]))
    d = np.array([0.4, -0.2])
    m = inst.mu[:, 1]
    x = m + inst.sigma * rng.standard_normal((400_000, 2))
    logp = -((x - m) ** 2) / (2 * inst.sigma ** 2)
    logq = -((x - m - d) ** 2) / (2 * inst.sigma ** 2)
    assert np.mean(np.sum(logp - logq, axis=1)) == pytest.approx(inst.kl_alternative(d), rel=0.02)


# --- Theorem 3 / Corollary 4: simulation vs formula ----------------------------
def test_ucb_counts_and_burdens_approach_formula():
    """Moderate-gap instance: UCB's E[n_a]/log T and E[R_i]/log T are within a
    finite-T band of c_min and D c_min. The band is wide because UCB's
    log f(t) = log t + 2 log log t term converges slowly (paper, Section 6)."""
    mu = np.array([[1.0, 1.6, 0.0],
                   [1.0, 0.2, 0.5],
                   [1.0, 0.6, 1.4]])
    inst = Instance(mu=mu, sigma=np.ones(3))
    T = 200_000
    counts = run_base(inst, T, [T], R=12, algo="ucb", seed=11)[0]
    ratio = counts[:, inst.suboptimal].mean(0) / (inst.c_min * math.log(T))
    assert np.all((ratio > 0.8) & (ratio < 1.7)), ratio
    reg = per_person_regret(inst, counts).mean(0) / math.log(T)
    target = inst.burden()
    # Per-person targets can be small through cancellation, so measure the
    # error against the scale of the individual terms, sum_a |delta_ia| c_a.
    scale = np.abs(inst.D) @ inst.c_min
    assert np.all(np.abs(reg - target) < 0.7 * scale), (reg, target, scale)


def test_blog_example_burden_ordering_in_simulation():
    """The person who loses from the near-tie pays; the others gain."""
    inst = blog_example()
    counts = run_base(inst, 50_000, [50_000], R=10, algo="ts", seed=5)[0]
    reg = per_person_regret(inst, counts).mean(0)
    assert reg[2] > 0 > reg[0]
    # person 3 pays more than the whole collective's total regret
    assert reg[2] > reg.sum()


# --- Theorem 5 (cone): the forced-exploration construction --------------------
@pytest.mark.parametrize("g", [np.array([0.0, 3.0, 0.5]), np.array([10.0, 0.0, 40.0, 2.0])])
def test_forced_schedule_within_proof_bounds(g):
    """f_a(T) in [g_a log(T - M), g_a log T + 1] with M = log 2 * sum g + K, for
    all T beyond the threshold used in the proof."""
    T_max = 20_000
    sched = forced_schedule(g, T_max)
    K = len(g)
    M = math.log(2) * g.sum() + K
    f = np.zeros(K)
    for t in range(1, T_max + 1):
        if sched[t - 1] >= 0:
            f[sched[t - 1]] += 1
        if t > 2 * (g.sum() * math.log(t) + K):  # threshold from the proof
            assert np.all(f <= g * math.log(t) + 1 + 1e-9)
            assert np.all(f >= g * math.log(t - M) - 1e-9), (t, f)


def test_forced_planner_shifts_burden_by_D_g():
    """E[R_forced(T)] - E[R_base(T)] ≈ D g log T at finite T (the base planner
    runs on its own samples, so the shift does not depend on its convergence)."""
    inst = compensable_example()
    g = np.array([0.0, 0.0, 0.0, 300.0])
    T = 100_000
    base = per_person_regret(inst, run_base(inst, T, [T], R=16, algo="ucb", seed=21)[0])
    forced = per_person_regret(inst, run_forced(inst, g, T, [T], R=16, algo="ucb", seed=22)[0])
    diff = (forced.mean(0) - base.mean(0)) / math.log(T)
    se = np.sqrt(forced.var(0, ddof=1) / 16 + base.var(0, ddof=1) / 16) / math.log(T)
    pred = inst.delta @ g
    assert np.all(np.abs(diff - pred) < 4 * se + 0.05 * np.abs(pred)), (diff, pred, se)


def test_forced_planner_is_consistent_on_other_instance():
    """The forced rates are tuned to one instance; on another instance the
    planner must still pull suboptimal arms only O(log T) times."""
    g = np.array([0.0, 0.0, 0.0, 300.0])
    other = Instance(mu=np.array([[0.2, 1.0, 0.0, 0.1],
                                  [0.1, 0.9, 0.3, 0.0],
                                  [0.0, 1.1, 0.2, 0.2]]), sigma=np.ones(3))
    cps = [10_000, 100_000]
    counts = run_forced(other, g, 100_000, cps, R=6, algo="ucb", seed=3).mean(1)
    sub = other.suboptimal
    per_log = counts[:, sub] / np.log(cps)[:, None]
    bound = other.c_min * 2.0 + g[sub] + 50
    assert np.all(per_log < bound)


# --- Section 5: LPs -----------------------------------------------------------
def test_minmax_matches_random_search():
    """The LP value is no worse than a random search over c >= c_min."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        inst = random_instance(rng, N=4, K=3, min_gap=0.2)
        lp = minmax(inst)
        best = inst.burden().max()
        for _ in range(20_000):
            c = inst.c_min + rng.exponential(scale=inst.c_min.mean(), size=inst.c_min.shape) \
                * (rng.random(inst.c_min.shape) < 0.5)
            best = min(best, inst.burden(c).max())
        assert lp.value <= best + 1e-6
        assert lp.value >= best - 0.05 * abs(best) - 1e-6


def test_ville_certificate_iff_no_improvement():
    """Two independent LPs agree: a certificate exists iff the min-max value
    equals the utilitarian planner's worst burden."""
    rng = np.random.default_rng(1)
    agree = 0
    for _ in range(200):
        inst = random_instance(rng, N=4, K=4, min_gap=1e-6)
        m = minmax(inst)
        umax = inst.burden().max()
        improvable = m.value < umax - 1e-7 * max(1.0, abs(umax))
        certificate = ville_certificate(inst) is not None
        assert improvable != certificate
        agree += 1
    assert agree == 200


def test_minmax_lower_bounds():
    rng = np.random.default_rng(2)
    for _ in range(50):
        inst = random_instance(rng, N=5, K=4, min_gap=1e-6)
        m = minmax(inst)
        assert m.value >= inst.total() / inst.N - 1e-7
        nonneg = np.all(inst.D >= 0, axis=1)
        if nonneg.any():
            assert m.value >= inst.burden()[nonneg].max() - 1e-7


def test_budget_feasibility_threshold():
    inst = compensable_example()
    m = minmax(inst)
    assert budget_feasible(inst, np.full(inst.N, m.value + 1e-6))
    assert not budget_feasible(inst, np.full(inst.N, m.value - 1e-3))
    # irreducible burden: in the blog example person 3 weakly prefers a* to all
    blog = blog_example()
    b3 = blog.burden()[2]
    beta = np.array([1e9, 1e9, b3 - 1e-3])
    assert not budget_feasible(blog, beta)


def test_compensable_example_numbers():
    """Hand-computed values from the paper's Example 2 (sigma_i = 1)."""
    inst = compensable_example()
    m = minmax(inst)
    assert m.value == pytest.approx(256 / 3, rel=1e-6)          # 85.33
    assert m.c[-1] == pytest.approx(960, rel=1e-6)
    assert m.price == pytest.approx(162, rel=1e-6)


# --- Lemma 2 with a common shock ----------------------------------------------
@pytest.mark.parametrize("seed", range(3))
def test_kl_minimisation_general_covariance(seed):
    """With covariance diag(sigma^2) + tau^2 11', the cheapest shift of total
    Delta costs Delta^2 / (2 * 1'Sigma 1), at d = Sigma 1 Delta / (1'Sigma 1)."""
    rng = np.random.default_rng(seed)
    N = 5
    inst = Instance(mu=rng.normal(size=(N, 2)), sigma=rng.uniform(0.3, 2.0, N), tau=rng.uniform(0.2, 1.0))
    Delta = rng.uniform(0.1, 2.0)
    cons = {"type": "eq", "fun": lambda d: np.sum(d) - Delta}
    res = minimize(inst.kl_alternative, x0=np.full(N, Delta / N), constraints=[cons],
                   method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    one = np.ones(N)
    s2 = one @ inst.Sigma @ one
    assert s2 == pytest.approx(inst.sigma2)
    assert res.fun == pytest.approx(Delta ** 2 / (2 * s2), rel=1e-6)
    np.testing.assert_allclose(res.x, inst.Sigma @ one * Delta / s2, rtol=1e-3, atol=1e-6)


def test_burden_scales_inversely_with_population_only_without_common_shock():
    """Replicating the population m times divides burdens by m when noise is
    idiosyncratic, and leaves them of order one when there is a common shock."""
    base = blog_example()
    m = 10
    big = Instance(mu=np.tile(base.mu, (m, 1)), sigma=np.tile(base.sigma, m))
    np.testing.assert_allclose(big.burden()[:3], base.burden() / m, rtol=1e-9)
    shock = Instance(mu=base.mu, sigma=base.sigma, tau=1.0)
    big_shock = Instance(mu=np.tile(base.mu, (m, 1)), sigma=np.tile(base.sigma, m), tau=1.0)
    ratio = big_shock.burden()[2] / shock.burden()[2]
    assert 0.7 < ratio < 1.0


# --- time-sharing and the status quo -----------------------------------------
def test_minmax_conflict_blog_example_and_random_search():
    r = minmax_conflict(blog_example().mu)
    np.testing.assert_allclose(r.p, [0.6875, 0.3125, 0.0], atol=1e-6)
    assert r.value == pytest.approx(0.34375, abs=1e-7)
    rng = np.random.default_rng(0)
    mu = rng.normal(size=(4, 5))
    best = minmax_conflict(mu).value
    ps = rng.dirichlet(np.ones(5) * 0.5, size=20000)
    assert all(conflict(mu, p).max() >= best - 1e-9 for p in ps[:2000])
    assert min(conflict(mu, p).max() for p in ps) < best + 0.15


def test_ir_optimum_status_quo_example():
    mu = status_quo_example().mu
    r = ir_optimum(mu, 3, 0.0)
    np.testing.assert_allclose(r.p, [0.6, 0.4, 0, 0], atol=1e-8)
    assert r.welfare == pytest.approx(2.96)
    assert r.duals.sum() == pytest.approx(0.2)        # price of protecting persons 1 and 2
    # the dual is the derivative of welfare in the tolerance
    eps = 1e-3
    assert (ir_optimum(mu, 3, eps).welfare - r.welfare) / eps == pytest.approx(0.2, rel=1e-6)


def test_revealed_weights_exist_iff_no_unanimous_improvement():
    """Weights rationalising a0 exist exactly when no time-share makes
    everyone strictly better off than a0 (theorem of the alternative)."""
    from scipy.optimize import linprog
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(200):
        N, K = 3, 4
        mu = rng.normal(size=(N, K))
        a0 = int(rng.integers(K))
        # largest s with mu p >= mu_a0 + s
        A = np.hstack([-mu, np.ones((N, 1))])
        r = linprog(np.r_[np.zeros(K), -1.0], A_ub=A, b_ub=-mu[:, a0],
                    A_eq=np.r_[np.ones(K), 0.0][None, :], b_eq=[1.0],
                    bounds=[(0, None)] * K + [(None, None)], method="highs")
        improvable = r.x[-1] > 1e-8
        w = revealed_weights(mu, a0)
        assert (w is None) == improvable
        if w is not None:
            assert np.all(w @ mu <= w @ mu[:, a0] + 1e-7)
        seen.add(improvable)
    assert seen == {True, False}


def test_pc_planner_never_breaches_the_floor():
    inst = status_quo_example()
    for seed in range(3):
        rec, U, p, worst = run_pc(inst, 3, 0.05, 1.0, 20000, seed=seed)
        assert worst >= -0.05 - 1e-9
        assert np.all(U >= -0.05 * rec[:, None] - 1e-6)
        assert p[3] < 0.95                           # it does leave the status quo


# --- the one-reform toy -------------------------------------------------------
def test_reform_thresholds():
    # one block and no option to learn: try the reform iff its prior mean is positive
    assert reform.solve(4, 3.0, 0.0, 4).threshold() == pytest.approx(0.0, abs=0.011)
    # with a switching cost c the reform must repay it: 4 m > c
    assert reform.solve(4, 3.0, 2.0, 4).threshold() == pytest.approx(0.5, abs=0.011)
    # option value: with learning the planner tries reforms it expects to be worse,
    # more so at longer horizons; frictions raise the bar
    th = [reform.solve(T, 3.0).threshold() for T in (4, 20, 40)]
    assert th[0] < 0 and th[0] > th[1] > th[2]
    assert reform.solve(20, 3.0, 2.0, 1).threshold() > th[1]
    assert reform.solve(20, 3.0, 0.0, 4).threshold() > th[1]


def test_reform_bad_reforms_are_dropped_sooner():
    plan = reform.solve(40, 3.0)
    r = [plan.simulate(0.0, th, paths=2000).mean() for th in (-1.0, 0.0, 1.0)]
    assert r[0] < r[1] < r[2]


# --- institutions for the scorecard -------------------------------------------
def test_institutions_scores_are_time_averages_of_arm_rewards():
    """Every planner's average reward vector is a mixture of the columns of mu,
    the utilitarian learner improves with the horizon, and the individually
    rational planner stays within its tolerance of the status quo."""
    from regvec.institutions import run_index, run_pilot, run_ir
    inst, hs = status_quo_example(), [50, 500, 5000]
    W = inst.welfare
    for avg in (run_index(inst, 3, hs, R=50, algo="ucb"), run_index(inst, 3, hs, R=50, algo="ts"),
                run_pilot(inst, 3, hs, R=50)):
        w = avg.sum(axis=2)
        assert np.all(w <= W.max() + 1e-9) and np.all(w >= W.min() - 1e-9)
    ts = run_index(inst, 3, hs, R=200, algo="ts").sum(axis=2).mean(axis=1)
    assert ts[0] < ts[1] < ts[2] and ts[2] > 2.95
    ir = run_ir(inst, 3, 0.05, 1.0, hs, R=3)
    assert np.all(ir - inst.mu[:, 3] >= -0.05 - 1e-9)


def test_pooled_planner_respects_the_welfare_floor_and_beats_per_person_floors():
    from regvec.institutions import run_pooled, run_ir
    inst, hs = status_quo_example(), [3000]
    w, worst = run_pooled(inst, 3, 0.05, 1.0, hs, R=4)
    assert worst >= -3 * 0.05 - 1e-9
    ir = run_ir(inst, 3, 0.05, 1.0, hs, R=4).sum(axis=2)
    assert w.mean() > ir.mean()


def test_certification_bounds():
    """Pooling can only lower the number of trials needed (independent noise),
    and the per-person bound blows up as the marginal loser's slack vanishes."""
    from regvec.institutions import certification_pulls
    rng = np.random.default_rng(0)
    for _ in range(200):
        N = int(rng.integers(2, 8))
        gain, sig = rng.normal(size=N), rng.uniform(0.3, 2.0, N)
        per, pooled = certification_pulls(gain, sig, alpha=float(rng.uniform(0.05, 3.0)), q=float(rng.uniform(0.1, 1)))
        if np.isfinite(per):
            assert pooled <= per * (1 + 1e-12)
    g = np.array([-0.2, -0.2, 0.8])
    a, _ = certification_pulls(g, np.ones(3), 0.05, 0.2)
    b, _ = certification_pulls(g, np.ones(3), 0.05, 0.24)
    assert b > 10 * a and certification_pulls(g, np.ones(3), 0.05, 1.0)[0] == float("inf")
