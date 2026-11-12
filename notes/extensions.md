# Regret is a vector — extension ideas

These are collected from the blog drafts (`regret-is-a-vector`, `who-pays-for-exploration`) and
the 2026-11-11 review. None of them is in the core paper. Each one is a possible section,
follow-up paper or open-problem entry.

**Core paper scope**
- Public arms.
- Utilitarian target.
- Regret measured against a*.
- Asymptotic (log T).
- Main result: the cone of achievable per-person burdens, plus the irreducible burden and the min-max LP.

## Change the baseline or the objective

1. **Status-quo baseline.** Measure R_i against the incumbent policy a_0 rather than a*.
   - This asks: who was made worse off by trying anything at all? It is probably closer to what people mean when they complain about reform.
   - It links to *conservative bandits* (Wu, Shariff, Lattimore & Szepesvári 2016), which keep cumulative reward above the baseline's. A per-person version would be a "no one worse off than under the status quo" constraint.
2. **Weighted welfare.** W_w = Σ w_i μ_i.
   - Changing the weights changes both Δ_a and the shares κ, and so changes which policies count as near-ties and who pays.
   - Question: can the weights be chosen to equalise burden, and at what cost in utilitarian regret?
3. **Non-linear targets: max-min, Nash welfare, prioritarian.** These break the linear decomposition.
   - Question: what is the forced burden when a planner learns the argmax of a non-linear social welfare function (SWF)?
   - Hossain, Micha & Shah (2021) and Barman et al. study Nash regret, but they collapse it to a scalar.
4. **Which target makes the burden fairest?** Treat the choice of SWF as part of the design problem: a meta-level trade-off between learning cost and the distribution of burden.

## Change the structure of the arms

5. **Private ↔ public spectrum.** Interpolate between Baek & Farias (private arms, choose who explores) and this paper (public arms).
   - Example: policies that are partly individualisable, such as opt-outs or exemptions.
   - Question: how much individualisability is needed before the burden can be equalised?
6. **Partitioning / federalism.** Split the population into G groups, each running its own bandit. A group's members only pay for policies tried on them, but each group learns more slowly.
   - Question: what is the optimal partition, with and without information sharing between groups (clustering-of-bandits / multitask bandits)?
   - This is the "laboratories of democracy" trade-off made formal.
7. **Structured arms (Graves–Lai).** If preferences are low-dimensional or arms are correlated, individual observations do carry information beyond the welfare sum.
   - Questions: how does the forced burden change, and does structure let you rule out contested arms more cheaply?
8. **Irreversible arms.** Some arms are absorbing or catastrophic, so the setting is no longer a stationary bandit; this is the safe-exploration setting.
   - This links to the offline-pessimism vs online-optimism point in the essay.
   - Question: the right amount of optimism as a function of reversibility.

## Change the horizon or the information

9. **Finite T and Bayesian versions.** Real policy runs for a handful of rounds.
   - At small T, the main question is which arms get tried at all.
   - Approaches: Bayesian regret, Gittins indices, and per-person burden under Thompson sampling at finite T.
10. **Non-stationarity.** Preferences drift, and institutions shape preferences (the formation layer in the framework post).
    - Question: how does the burden change when trying a policy changes what people want?
11. **Unknown or heterogeneous noise, non-Gaussian rewards.** Generalise the KL computation (exponential families; unknown σ_i).

## Constraints, rights and compensation

12. **Per-person budgets as rights.** Treat this as a constrained bandit with R_i ≤ B_i.
    - Already shown: budgets below the irreducible burden are infeasible.
    - Open: the optimal planner when they are feasible, and what gives way when they are not (stop learning, change the objective, or foreclose).
13. **Foreclosure ↔ minimality.** Forbidding a contested arm removes its burden but risks deleting the true optimum.
    - The minimality argument says uncertainty is a reason *not* to forbid, so this is a direct tension with a "right not to be experimented on".
    - Plan: formalise the joint problem of what to forbid and what to explore.
14. **Compensation and transfers.** If utility is transferable, only the total matters, so the who-pays problem *is* the failure of transferability.
    - Questions: what partial transfers would close the gap? What are the costs of finding the losers, and of delayed compensation (insurance, social safety nets, "flexicurity")?
15. **Equipoise and stopping rules.** Clinical trials randomise only under expert uncertainty, and stop early once one arm is clearly better.
    - Plan: formalise this as best-arm identification with per-person caps, then compare it with legislation, which has no data-monitoring board.
16. **Making the explorer pay.** Liability, bonds and insurance, i.e. who funds the burden when the explorer is a firm (AI developer) rather than the state.

## Strategic behaviour

17. **Exit.** The losers from a contested arm move away (Tiebout), which changes the population and therefore the gaps.
18. **Voice and misreporting.** Losers exaggerate their losses so that the contested arm is ruled out faster. Question: is there an incentive-compatible version?
19. **Refusal to explore.** Incentivised exploration (Kremer, Mansour & Perry 2014; Mansour, Slivkins & Syrgkanis) combined with per-person accounting. Each participant who explores is conscripted behind an information veil: whose burden is that?
20. **Information externality.** The explorer produces information everyone gets for free (Bolton & Harris; Keller, Rady & Cripps), so there is too little exploration.
    - Question: how do you net the imposed risk against the gifted information?
    - Raghavan et al. (COLT 2018) on the externalities of exploration are relevant here.

## Applications and empirics

21. **AI deployment.**
    - Staged rollouts, limited releases, regulatory sandboxes and A/B tests on users all explore on a public arm.
    - Question: who is the test population, and does the burden fall on a predictable group (early adopters, particular regions or languages)?
    - This is the main AI-governance hook.
22. **Policy pilots and case studies.**
    - NZ 1984–93 reforms: burden concentrated on particular towns and industries.
    - Pre-1986 NZ homosexuality law: the status quo as an experiment whose costs fall on one group.
    - Sunset clauses, including the symmetric idea of sunsetting *old* laws.
23. **Visibility as the mechanism.** Many remedies may work mainly by making the regret vector visible: who is in the experiment, what it is costing them, and when it ends. That is a measurement and transparency proposal, and possibly an empirical one.

## Gaps in the current draft to fix first

These are not extensions.
- Write out the change-of-measure proof rigorously.
- State the "forced" corollary correctly. It is exact only for asymptotically optimal planners. For consistent planners it gives the cone, and the irreducible burden applies only to people who weakly prefer a*.
- Cite and contrast Baek & Farias (NeurIPS 2021 / Management Science 2024).
