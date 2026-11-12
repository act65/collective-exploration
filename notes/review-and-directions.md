# Review of the draft, and where to take it (2026-11-12)

Companion to `lit-review.md`. Part 1 evaluates the draft as it stands. Part 2 answers the
questions about fairness, privilege, frictions and distortion. Part 3 proposes a restructured
paper. Statements marked **(conjecture)** are not proved.

## 1. Evaluation of the draft

### What holds up

I re-derived every proof. They are correct as stated; the 17 tests pass.

- Lemma 1, Theorem 2, Theorem 4: correct. The Lattimore–Szepesvári Theorem 8.1 bound quoted in
  Proposition 3 matches my memory of the book; still check against the page.
- Theorem 5 (cone): correct, including the schedule lemma.
- Proposition 10: correct. It can be made self-contained by deriving it from Gordan's theorem
  applied to `[D_I; -I]`, which removes the dependence on finding "Ville" in Mangasarian.

### Six problems, most serious first

**1. The log T burden is second order next to a linear term the paper never mentions.**
In Example 1, persons 1 and 2 prefer `b` to `a*` by 0.5 per round. Living under the utilitarian
choice costs each of them `0.5 T`. At `T = 10^6` that is 500,000, against a learning burden of
`661 log T ≈ 9,100` for person 3. The paper studies the smaller quantity. This is defensible
(the learning cost is the part due to ignorance), but it has to be said, and the full
decomposition (§2.1 below) should be in the paper. A referee will say it otherwise.

**2. "Fairness is free" is the wrong name.** Proposition 10 says the utilitarian planner cannot
be *improved* in the min-max sense. In Example 1 that "free" outcome is (−299, −299, +661). The
honest reading of the 79–93% figure is "in most instances, fairness through exploration is
*unavailable*". The abstract's last sentence already says this; the terminology should match.

**3. The regime does not match the motivation.** Convergence needs `T ≈ 10^6` in the paper's
own example. Governments get perhaps tens of rounds per policy area. Every result is about a
regime real policy never reaches. The limitations section admits this; it is the main reason
to add a finite-horizon toy (§2.3).

**4. Per-person burdens vanish as 1/N under the paper's own noise model.** Replicate the
population `m` times: `Delta` and `sigma^2` both scale by `m`, so `c_min` and every `b_i` scale
by `1/m`. With independent noise, a country of five million learns in one round. What makes
policy learning slow is *common* shocks. The fix is easy: Lemma 1 holds for any noise
covariance `Sigma`, with `sigma^2` replaced by `1' Sigma 1` (minimise `d' Sigma^{-1} d / 2`
subject to `1'd = Delta`; the minimiser is `Sigma 1 Delta / (1' Sigma 1)`). With a common shock
of variance `tau^2`, `1' Sigma 1 = sum sigma_i^2 + N^2 tau^2`, and per-person burdens are O(1).
State the lemma that way and add a remark on scaling.

**5. The claim "the only lever is how often each policy is tried" is too strong.** It is true
only among planners consistent *for the utilitarian arm*. Hossain et al. use the same model and
get fairness by **time-sharing**: target a distribution over arms. That is a linear-in-T lever
and it is the standard one in social choice (taking turns). The paper should present its
result as: holding the utilitarian target fixed, the lever is weak.

**6. Positioning.** The shared-arm model is Hossain et al.'s; the "achievable set, then pick a
fair point" programme is Baek & Farias's; min-max burden is Blanchard & Goyal's. The draft's
"as far as we know, new" needs these three contrasts. Jung, Kannan & Lutz and Strulovici (2010)
are missing from the bibliography.

### Smaller points

- A reinterpretation worth adding: `delta_ia` is both what person `i` loses while `a` is tested
  *and* what `i` gains from `a` being ruled out. So the people who pay to rule out a policy are
  exactly those who benefit from its being ruled out. This is the benefit principle of public
  finance, and it cuts against reading the burden as simply unjust.
- Individual observations do not help find `a*`, but they are **necessary to know who is
  paying**: the matrix `D` cannot be estimated from the welfare signal. Any fair planner needs
  them. This turns Remark 3 from a negative result into a two-sided one.
- A missing quantity: the transfer volume that would equalise burdens, `(1/2) sum_i |b_i − mean(b)|`.
  In Example 1 it is 640 per `log T` against a total regret of 64. Still only logarithmic,
  which is the quantitative case for compensating the losers of experiments (§2.5).
- The paper had no figures. Four are now in: setup diagram, anatomy of the example, the cone,
  simulated trajectories (`code/scripts/figures.py`).

## 2. Directions

### 2.1 One decomposition to organise everything

Measure each person against their own favourite policy. Let `a_inf` be the policy the
institution converges to. Then

    T (max_a mu_ia) − sum_t mu_{i,a_t}
      =  T (max_a mu_ia − mu_{i,a*})        conflict:   others exist
       + T (mu_{i,a*} − mu_{i,a_inf})       distortion: the mechanism is coarse
       + sum_t (mu_{i,a_inf} − mu_{i,a_t})  learning:   nobody knew

The draft studies only the third term with `a_inf = a*`. The second term is exactly the
ballot-structure paper's distortion, `W(x*) − W(y(Q))`, kept per person. The first is the
price of sharing a policy at all. A status quo `a_0` adds a fourth reading: replace
`max_a mu_ia` by `mu_{i,a_0}` and the left side becomes "what reform cost me".

This gives the paper a spine: three sources of per-person loss, each with its own scale
(`T`, `T`, `log T`), its own remedy (time-sharing or exit; better ballots or richer feedback;
compensation or private arms), and one section each.

### 2.2 Privilege

Your proposal: the privileged are those with low regret under the status quo. Three ways to
make it precise, in increasing order of interest.

- **P1, absolute.** `rho_i = max_a mu_ia − mu_{i,a_0}`. Low `rho_i` means the status quo is
  close to your ideal.
- **P2, relative to reform.** `mu_{i,a_0} − mu_{i,a*} > 0`: you lose from the welfare-optimal
  change. These are the people for whom reform is not a Pareto improvement.
- **P3, revealed weights.** Find weights `w ≥ 0` such that `a_0` maximises
  `sum_i w_i mu_ia`. Privilege is the weight the status quo implicitly puts on you. If only
  very unequal weights rationalise `a_0`, the distance from uniform weights measures structural
  privilege in the population. This is an LP, and it is the "inverse-optimum welfare weights"
  idea from public economics applied to policies. The literature search found no formal
  status-quo-relative definition of privilege, so any of these would be new.

Things that then follow, or should:

- **Exploration is progressive.** If arms are exchangeable a priori, person `i`'s expected
  one-round gain from trying an untried policy is `m_i − mu_{i,a_0}`, with `m_i` their prior
  mean. It is negative exactly for people the status quo places above their own average. The
  interim cost of experimentation rises one-for-one with privilege. Short proof, and it
  explains why the comfortable are rationally conservative.
- **Selection makes it worse.** If `a_0` was chosen as the best of past policies for a subset
  `S`, then members of `S` sit at a maximum, and any exploration regresses them toward the mean.
- **The fully privileged cannot be compensated through exploration.** If `a_0` is your favourite
  policy, every trial costs you. This is Corollary 7 with the baseline changed.
- **Not exploring is not neutral.** Keeping `a_0` forever imposes `T (mu_{i,a*} − mu_{i,a_0})`
  on everyone the status quo underserves. Meyer's "A/B illusion" is the same point.
- **The price of a veto (conjecture).** Maximise welfare over distributions `p` on arms subject
  to `mu p ≥ mu_{·,a_0}` for every person. The dual variable on person `i`'s constraint is
  the welfare the collective forgoes per unit of protection for `i`. That is a measurable
  price of privilege, and it is zero for anyone whose constraint is slack.

### 2.3 Frictions: commitment, switching, few rounds

What the literature says first: switching costs do not change the asymptotic rate (Agrawal et
al. 1988), and `log log T` batches recover the minimax rate (Perchet et al.; Gao et al.). So
frictions are invisible in the draft's regime. They matter only at finite horizon, which is
another reason to add one.

Proposed toy, to sit beside the Gaussian example: **one known status quo, one unknown reform.**
Status-quo welfare `W_0` is known. The reform's welfare is `N(m, s^2)`, observed with noise.
The Bayes-optimal planner tries the reform if and only if its Gittins index exceeds `W_0`,
and abandons it when the posterior index falls below. Then:

- The rational planner tries reforms it *expects to be worse* (the index exceeds the mean).
  That is the option value of learning, and it is the answer to "what would a truly rational
  approach do".
- A switching cost widens the band: harder to start, slower to stop (hysteresis).
- A commitment period `L` (a term of office) lowers the option value, so fewer reforms clear
  the bar.
- Person `i` pays `(mu_{i,0} − mu_{i,1})` times the expected trial length, which is a hitting
  time. Everything is computable and plottable.

A per-person switching cost `s_i` (retraining, moving, relearning a product) fits the same toy
and gives a second vector alongside regret.

### 2.4 What "fair exploration" could mean, and a provable algorithm

| Notion | Definition | Status |
|---|---|---|
| Min-max burden | minimise `max_i (Dc)_i` over the cone | In the draft. Often vacuous (problem 2). |
| Individually rational | each person's cumulative reward stays above `(1 − alpha)` times the status quo's, at all times, with high probability | **Open, per the search. My first choice.** |
| Nash bargaining | maximise `prod_i (u_i − d_i)`, disagreement point = status quo | Baek & Farias's notion carried to public arms. |
| Benefit-proportional | burden share proportional to stake in the result | Close to what utilitarian learning already does (§1, smaller points). |
| Time-shared | target a fair distribution over arms (Nash welfare, leximin) | Hossain et al., scalar regret only. |

**Recommended theorem target: Pareto-conservative exploration.** Target the welfare-best
distribution over arms within the individually rational polytope. Use optimism for the
objective and pessimism for each person's constraint, as in safe linear bandits. Expected
result **(conjecture)**: Õ(√T) welfare regret against the best individually rational
distribution, with no person's constraint violated at any time with high probability. The
privileged are exactly those whose constraints bind, and §2.2's shadow prices come out of the
same LP. Per-person observations are essential here, which gives Remark 3 its second half.

**Second target: close the draft's open problem.** An adaptive planner that reaches the
min-max point without knowing the instance. OSSB or the Degenne–Shao–Koolen saddle-point
method should do it: estimate `D` and `Delta`, solve the LP at the estimate, track it. The LP
value is continuous in the instance even where the solution is not, which is what the argument
needs. I think this is provable at the `v* + epsilon` level **(conjecture)**.

### 2.5 Recommendations a government or firm could act on

Each is tied to a result, existing or proposed.

1. **Publish the regret vector.** Who is exposed, what it costs them, when it ends. The planner
   needs per-person data to know this (§1), so collecting it is not optional.
2. **Compensate, don't give vetoes.** The transfer needed to equalise learning burdens is
   logarithmic in the horizon; the gains from finding the right policy are linear. Anesi &
   Bowen reach the same conclusion in a strategic model.
3. **Privatise the arm where the contest is close and polarising.** Those are the policies that
   must be tried longest and whose losers pay most. Pilots, staged rollouts, opt-outs.
4. **Treat the status quo as an arm.** Declining to experiment has a regret vector too, and it
   falls on the people the status quo serves worst.
5. **Experiment more where reversal is cheap**, less where commitment is long (§2.3).
6. **Firms differ from states in one way that matters:** customers can exit. Exit caps each
   person's burden but changes the population the firm learns from (Gieczewski & Kosterina).

### 2.6 Distortion and the ballot-structure paper

Two bridges, the second more ambitious.

- **Feedback.** Replace cardinal rewards with votes (approve / disapprove against the status
  quo). The planner then converges to a majority winner, not `a*`, and the gap is distortion,
  linear in `T`. A few cardinal queries shrink distortion sharply in the static setting
  (Amanatidis et al.); the dynamic question is how exploration and elicitation trade off. The
  search found no paper defining per-voter regret under ordinal feedback.
- **Structure.** Let an arm be a bundle of issue decisions `x in {0,1}^m` with the
  ballot paper's dependency graph. This is a structured bandit, the case Remark 3 sets aside.
  Separable issues can be explored independently; coupled issues must be explored jointly. The
  dependency graph would then determine both the distortion of a ballot and the cost and
  incidence of learning. That is a second paper, but a paragraph belongs in this one.

## 3. Suggested shape of the paper

1. Introduction, with the setup figure and the three-term decomposition.
2. Setting: general noise covariance, a status quo `a_0`, the Gaussian running example.
3. Who pays for learning (current sections 3–6, tightened; rename "fairness is free").
4. Who pays for the status quo: privilege, exploration is progressive, the price of a veto.
5. A fair planner: Pareto-conservative exploration, or the adaptive min-max planner.
6. Finite horizon and frictions: the one-reform toy.
7. Distortion: what changes when the planner only sees votes.
8. Recommendations.

Sections 4 and 5 are where new theorems are needed. Section 6 is a worked example, and 7 can be
a short section pointing forward.

## Decisions for you

- **Scope.** One paper with all of §3, or a tight theory paper (sections 1–5) plus a second on
  distortion and ballots?
- **Fairness notion.** Individually rational (my recommendation) or min-max as the headline?
- **Venue.** EC / NeurIPS want sections 3 and 5 sharp. A governance or economics venue wants 4,
  6 and 8.
