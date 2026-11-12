# Regret is a vector

*Who pays for exploration when a collective learns a public policy?*

A planner tries policies to learn which is best for total welfare. The policy applies to everyone at
once (a **public arm**), so each person bears a different share of the cost of learning, and regret
becomes a vector. The paper characterises that vector: what is forced on each person, what any
consistent planner can change, and when fairness is free.

Developed from the blog drafts `act65.github.io/_drafts/governance-mechanisms/2026-10-24-regret-is-a-vector.md`
and `_drafts/inbetween-posts/2026-10-24-who-pays-for-exploration.md`.

## Layout

- `paper/` — the paper (`make` builds `main.pdf`; `make check` lists warnings).
- `paper/sections/` — the time-sharing, status-quo and finite-horizon sections.
- `code/regvec/` — closed forms (`instance.py`), planners vectorised over seeds (`planners.py`), LPs over the cone (`fairness.py`), time-sharing and status-quo LPs (`sharing.py`), the individually rational planner (`safe.py`), the one-reform dynamic program (`reform.py`).
- `code/tests/` — 28 tests (`cd code && python -m pytest -q`, about 30 s). Each test compares a simulation or an independent computation with a closed form.
- `code/scripts/simulate.py` — numbers for the asymptotic sections. A full run takes about 4–5 min; `--quick` is faster. Output goes to `code/reports/results.json`, `code/reports/simulate.log` and `paper/figures/`.
- `code/scripts/extensions.py` — numbers for the time-sharing, status-quo and finite-horizon sections (about 2 min), written to `code/reports/extensions.json`.
- `code/scripts/scorecard.py` — the finite-horizon scorecard of institutions (about 2 min), written to `code/reports/scorecard.json`; the planners are in `code/regvec/institutions.py`.
- `code/scripts/figures.py` — all figures in the paper, drawn from closed forms and the two reports.
- `notes/extensions.md` — the list of extensions and open problems.
- `notes/review-and-directions.md`, `notes/lit-review.md` — the 2026-11-12 review of the first draft and the literature search.

## Status (2026-11-12)

**Added on 2026-11-12** (sections 7–10 of the paper).

| Result | Statement | Status |
|---|---|---|
| Lemma 1, general covariance | The lower bound uses σ² = 1ᵀΣ1, so common shocks are covered. With independent noise burdens fall as 1/N; with a common shock they do not. | Proved; tested. |
| Lemma (alternative) | Replaces the citation of Ville's theorem with a short derivation from Gordan's theorem. Used three times. | Proved. |
| Corollary 7, second half | A burden is irreducible exactly when a* is that person's favourite policy. | Proved (one line). |
| Proposition (time-sharing) | Smallest largest conflict loss is an LP; a* attains it iff the same alternative condition holds for the people with the largest conflict loss. | Proved; tested. |
| Proposition (no free trial) | A planner that tolerates no expected loss for anyone never leaves the status quo. | Proved. |
| Proposition (unanimity) | No unanimous improvement exists iff some welfare weights make the status quo optimal. | Proved; tested. |
| Proposition (safety) | The individually rational planner keeps every person within α of the status quo in every round, with confidence 1−δ. | Proved; tested. **No regret bound.** |
| Scorecard (section 10) | Seven institutions on the status-quo example, T = 30 to 30,000: welfare and each person's gain over the status quo. | Simulation only, one instance, untuned planners. |
| One-reform toy | Bayes-optimal thresholds and trial lengths with switching costs and terms. | Numerical only. |

"Fairness is free" is now "unimprovable" throughout: the condition says the largest burden cannot be reduced, not that the burdens are fair.

## Status of the first draft (2026-11-11)

**Proved in the paper, with full proofs.** Steps marked `% CHECK` in `main.tex` should be re-checked against the cited sources.

| Result | Statement | Notes |
|---|---|---|
| Lemma 1 | The cheapest perturbation that makes an arm look optimal costs Δ²/(2σ²), with σ² = Σσ_i². | |
| Theorem 2 | Lower bound with full vector observations: every consistent planner has liminf E[n_a]/log T ≥ 2σ²/Δ_a². | Individual observations do not help in the unstructured class. |
| Proposition 3 | UCB on the welfare sum attains the bound. | Cited from Lattimore & Szepesvári Thm 8.1. `% CHECK` the constants. |
| Theorem 4 | Exact per-person limit for every asymptotically optimal planner: b_i = 2σ² Σ_a δ_ia/Δ_a². | |
| Theorem 5 (cone) | The finite limit points of E[R(T)]/log T over consistent planners are exactly {Dc : c ≥ c_min}. Every point is attained by a deterministic forced-exploration planner, which depends on ν but is consistent everywhere. | Lemma 6 (the schedule keeps pace) is proved, and its bounds are checked in the tests. |
| Corollary 7 | Irreducible burden for anyone who weakly prefers a* to every arm. Anyone else's burden is unbounded below in the cone. | |
| Corollary 8 | A per-person budget is feasible iff an LP is. Fixed (non-log) caps are infeasible for people with b_i > 0. | |
| Propositions 9–10 | Min-max burden is an LP. The utilitarian planner is unimprovable iff some weighted average of the worst-off people weakly prefers a* to every arm. | Now derived from Gordan's theorem. |

**Corrections to the blog draft.**
- The blog's "forced" corollary ("any learner that eventually finds the best total has to impose it") is true only for *asymptotically optimal* planners.
- For merely consistent planners the achievable set is the cone. The burden is irreducible only for people who weakly prefer a*. Everyone else can be compensated, but only through arms they prefer to a*, and only at a utilitarian cost.

**Checked numerically** (`code/reports/simulate.log`, 20 seeds, T up to 10⁶).

*Blog example.*

| Planner | Person 3's burden / log T at T = 10⁶ | Formula | Comment |
|---|---|---|---|
| UCB | 671 ± 45 | 661 | Converges, but only by T ≈ 10⁶ (near-tie Δ/σ ≈ 0.06). |
| Thompson sampling | 273 ± 33 | 661 | Still about 40% of the asymptotic count. |

The ratio between people (≈ 2.2) is right at every horizon for both planners.

*Cone construction.* Forced planner minus base planner should shift the burden by Dg = (324, 324, −486). Observed at T = 10⁶:
- UCB base: (354, 354, −553);
- Thompson base: (313, 313, −460).

Both are within about one standard error.

*Random instances* (μ ~ N(0,1), 400 per size):
- The utilitarian planner is unimprovable in 79–93% of instances.
- Otherwise, min-max lowers the worst burden by a median of only 2–8%, and costs a median of 49–73% extra total regret.

**Open.**
- A regret bound for the individually rational planner.
- The per-person cost of learning a time-share.
- Finite horizon with several untried reforms.
- An *adaptive* planner that reaches the min-max point on every instance without knowing ν (compare Baek & Farias's PF-UCB).
- Finite-T theory.
- The extensions in [`notes/extensions.md`](notes/extensions.md):
  - status-quo baseline;
  - weighted or non-linear welfare;
  - partitioning / federalism;
  - finite-T and Bayesian versions;
  - strategic exit and voice;
  - structured arms (Graves–Lai);
  - transfers;
  - foreclosure vs minimality.

**Bibliography.** The five entries previously marked `% VERIFY` were checked on 2026-11-12. Entries now marked `% VERIFY` in `paper/references.bib` were added from memory and need volume and pages checked:
- Strulovici 2010;
- Fernandez & Rodrik 1991;
- Agrawal, Hegde & Teneketzis 1988;
- Perchet et al. 2016;
- Gao et al. 2019;
- Bird et al. 2016;
- Meyer et al. 2019;
- Procaccia & Rosenschein 2006.

Baek & Farias was checked against NeurIPS 2021 (pp. 22034–22045) and Management Science 70(12):8922–8939.
