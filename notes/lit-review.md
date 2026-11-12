# Literature review (2026-11-12)

Two web searches, one on bandit prior art and one on economics / ethics / social choice.
**[V]** = title, authors, venue and year confirmed by search this session. **[M]** = cited from
memory, exists but check volume and pages before it goes in the `.bib`. Regret rates quoted
for the friction papers are from memory; check them against the theorem statements before quoting.

## Novelty verdict

Nothing found pre-empts the per-person burden formula, the cone, or the Ville condition. A
negative search is not proof. Three things a referee will raise:

1. **The shared-arm model is not new.** Hossain, Micha & Shah (2021) and Jones, Nguyen &
   Nguyen (2023) have exactly our model (one arm applied to all N agents, means `mu[i,a]`). They
   target the Nash-welfare-optimal *distribution* over arms and report a scalar regret. What is
   new here is keeping the vector.
2. **Baek & Farias own the framing** "achievable set of per-group exploration costs over
   consistent policies; regret-optimal is unfair; pick a fair point". Their burden comes from
   *who pulls* a shared-mean arm; ours from heterogeneous losses under a common pull, where some
   people gain. They note their set need not be convex; ours is a cone. Say this explicitly.
3. **Blanchard & Goyal (2025) already minimise the max exploration burden across groups**, in a
   grouped model, non-asymptotically. Must be cited next to the min-max LP.

Results 1 and 2 of the draft (lower bound, exact burden) will read as easy corollaries. The
weight is on the cone, the LP, and the interpretation.

## A. Closest bandit work

| Paper | What it does | Relation |
|---|---|---|
| Hossain, Micha, Shah. "Fair Algorithms for Multi-Agent Multi-Armed Bandits." NeurIPS 2021 [V] | Same model. Learns the Nash-welfare-optimal distribution over arms, Õ(√T) regret. | Same setting, scalar regret, **time-sharing as the fairness lever** (see review §2). |
| Jones, Nguyen, Nguyen. "An Efficient Algorithm for Fair Multi-Agent Multi-Armed Bandit with Low Regret." AAAI 2023, 37(7):8159–8167 [V] | Improves the above to Õ(√(NKT)+NK). | Same. |
| Manupriya, Himanshu, Jagarlapudi, Ghalme. "Multi-agent MAB with Minimum Reward Guarantee Fairness." AAMAS 2025, arXiv:2502.15240 [V, author list from arXiv] | Welfare subject to each agent getting a fraction of its own best reward. | Per-agent guarantee on levels, not on exploration cost. Close to a per-person floor. |
| Caiata, Blair, Larson. "Procedural Fairness in Multi-Agent Bandits." arXiv:2601.10600 (2026) [V] | Equal decision share; fairness notions are mutually incompatible. | Normative neighbour. |
| Baek & Farias. "Fair Exploration via Axiomatic Bargaining." NeurIPS 2021; Mgmt Sci 70(12):8922–8939 [V] | Grouped bandit, one group arrives per round. Regret-optimal policies are arbitrarily unfair; Nash bargaining solution (PF-UCB); price of fairness. | Conceptual ancestor. PF-UCB is the template for an adaptive fair planner. |
| Blanchard & Goyal. "Collaborative Min-Max Regret in Grouped Multi-Armed Bandits." arXiv:2506.10313 (2025) [V] | Minimise max regret across groups; Col-UCB. | Nearest to our min-max objective. |
| Jung, Kannan, Lutz. "Quantifying the Burden of Exploration and the Unfairness of Free Riding." SODA 2020 [V] | A free rider gets O(1) regret, a self-reliant learner Ω(log t). | Precedent for "burden of exploration"; private arms. |
| Wang et al. "Exploration for Free: How Does Reward Heterogeneity Improve Regret in Cooperative Multi-agent Bandits?" UAI 2023 [V] | An arm optimal for one agent gives others free exploration. | Parallels our negative `delta_ia` (people who gain from exploration). |
| Xu & Klabjan. "Pareto Regret Analyses in Multi-objective MAB." ICML 2023 [V] | Scalar Pareto regret. | Vector rewards, scalar regret. |
| Busa-Fekete et al. "Multi-objective Bandits: Optimizing the Generalized Gini Index." ICML 2017 [V] | Fair aggregation of vector costs. | Fair outcomes, not fair exploration. |

## B. Tools for the open problem (adaptive fair planner)

- Combes, Magureanu, Proutiere, OSSB (NeurIPS 2017) [V]. Solve the lower-bound LP at the
  empirical instance, track it, with forced exploration. Simplest template: swap in the min-max LP.
- Degenne, Shao, Koolen. "Structure Adaptive Algorithms for Stochastic Bandits." ICML 2020 [V].
  Saddle-point iterates in place of an LP oracle. Our min-max is already a saddle point.
- Van Parys & Golrezaei. "Optimal Learning for Structured Bandits." Mgmt Sci 70(6), 2024 [V].
- Baek & Farias's PF-UCB is the same recipe with a different objective.

Per-person baselines: Wu et al. "Conservative Bandits" ICML 2016 [V] (scalar baseline);
Moradipari, Thrampoulidis, Alizadeh "Stage-wise Conservative Linear Bandits" NeurIPS 2020 [V];
Amani, Alizadeh, Thrampoulidis "Linear Stochastic Bandits Under Safety Constraints" NeurIPS 2019
[V]; Afsharrad, Moradipari, Lall "Multi-Agent Stage-wise Conservative Linear Bandits"
arXiv:2510.00602 [V]. **No work found on individually rational / Pareto-improving exploration
with per-person baselines under a shared arm.** That looks open.

## C. Political economy of experimentation

| Paper | Result | Relation |
|---|---|---|
| Fernandez & Rodrik (1991), AER 81(5) [M] | A reform that helps a majority ex post can be voted down ex ante when people don't know if they will win. | Status-quo bias from individual uncertainty. |
| **Strulovici (2010)**, "Learning While Voting", Econometrica 78(3) [M] | Public risky arm, heterogeneous voters learning their own types; majority voting under-experiments; unanimity can be worse. | **Closest economic precursor.** Strategic voters, no planner. |
| **Anesi & Bowen (2021)**, AEJ Micro 13(3) [V] | Without transfers experimentation is inefficient; small redistribution restores it under any rule without veto players. | **Compensation beats vetoes.** Central for the normative section. |
| Messner & Polborn (2012), J Pub Econ 96 [V] | With an option to wait, the optimal rule is a supermajority. | A justification for protecting the status quo. |
| Callander (2011) APSR [V]; Callander & Harstad (2015) QJE [M]; Cheng & Li (2019) AEJ Micro [V]; Volden, Ting & Carpenter (2008) APSR [M] | Policy search near the status quo; federal experimentation and free-riding. | Federalism / partition extension. |
| Majumdar & Mukand (2004) AER [M]; Dewatripont & Roland (1995) AER [M] | Reputation distorts starting and stopping experiments; gradualism keeps reversal cheap. | Commitment and irreversibility. |
| Gieczewski & Kosterina (2024) REStud [V] | With exit, pessimists leave and organisations over-experiment. | Firms (customers can exit) vs states. |

## D. Frictions

| Paper | Result | What it means for us |
|---|---|---|
| Agrawal, Hegde, Teneketzis (1988) IEEE TAC [M] | With switching costs, block allocation keeps the Lai–Robbins constant; switching regret is o(log T). | **Switching costs are asymptotically free.** Frictions only matter at finite T. |
| Banks & Sundaram (1994) Econometrica [M] | No index policy is optimal with switching costs. | The Bayesian structure breaks. |
| Dekel, Ding, Koren, Peres (2014) STOC [M] | Adversarial bandits with switching costs: Θ̃(T^{2/3}). | Rate changes only adversarially. |
| Perchet, Rigollet, Chassang, Snowberg (2016) Ann Stat [M]; Gao, Han, Ren, Zhou (2019) NeurIPS [M] | M batches: minimax regret ≈ T^{1/(2−2^{1−M})}; O(log log T) batches recover √T. | Terms of office are batches; only very few terms hurt. |
| Simchi-Levi & Xu (2019) NeurIPS [V] | Hard switching budget, phase transitions. | A cap on policy changes. |
| Farias & Madan (2011) Oper Res [V] | Discarded arms can't be recalled; bounded price of irrevocability. | Irreversible abandonment. |
| Joulani, György, Szepesvári (2013) ICML [V] | Delay is additive in stochastic problems. | Slow policy feedback. |

## E. Ethics of experimenting on people

- Bird, Barocas, Crawford, Diaz, Wallach (2016), "Exploring or Exploiting? Social and Ethical
  Implications of Autonomous Experimentation in AI", FAT/ML [V, check author list]. The informal
  statement of our question.
- Meyer et al. (2019) PNAS 116(22) [M]: people object to an A/B test of two policies they would
  accept untested. Meyer (2015) Colo Tech LJ [V]: an untested roll-out is an uncontrolled
  experiment. **The status quo is an arm.**
- Narita (2021) PNAS [V], EXAM: welfare-respecting randomisation (private arms).
- Kasy & Sautmann (2021) Econometrica [V]: adaptive experiments for policy choice. A correction
  to Theorem 1 was posted in 2021; cite with it.
- Abramowicz & Szafarz (2020) [V]; Picchio (2026) Econ & Phil [V]: equipoise in policy.
- Allen (2019) Geo Wash L Rev [V]: regulatory sandboxes.

## F. Privilege, baselines, fair public decisions

- **No established formal economic definition of privilege was found.** Nearest: Bothmann et al.
  (2025), "Privilege Scores", arXiv:2502.01211 [V], which is causal/counterfactual, not
  status-quo-relative. A status-quo-relative definition would be new.
- Baselines: Kaldor (1939), Hicks (1939), Scitovsky (1941) [M]; Nash (1950) disagreement point
  [M]; Buchanan & Tullock (1962), Rae (1969), May (1952) on unanimity and neutrality [M];
  Samuelson & Zeckhauser (1988) [M]; Roemer (1998) [M].
- Fair public decisions over time: Conitzer, Freeman, Shah (2017) EC [M]; Freeman, Zahedi,
  Conitzer (2017) IJCAI [V]; Lackner (2020) AAAI [V]; Fain, Goel, Munagala (2016) [V]. With
  known utilities, time-sharing a public arm gives proportional / Nash-fair outcomes.

## G. Distortion and learning

- Amanatidis, Birmpas, Filos-Ratsikas, Voudouris (2021), "Peeking Behind the Ordinal Curtain",
  Artificial Intelligence 296 [V]: a few cardinal queries per voter cut distortion sharply.
- Ebadian & Shah (2025) AAAI [V]; Mandal, Procaccia, Shah, Woodruff (2019) NeurIPS [V]:
  query / communication vs distortion frontiers.
- Urvoy et al. (2013), "Generic Exploration and K-armed Voting Bandits", ICML [V]; Zoghi et al.
  (2015) Copeland dueling bandits [V]: learning a voting winner from samples.
- Ahmed & Ghasemi (2026), arXiv:2605.01961 [V]: multi-user dueling bandits with Nash welfare.
- **Gap:** no paper found that defines regret under ordinal feedback from a population against a
  distortion benchmark, or distributes it per voter.

## Bibliography entries now verified

- Raghavan, Slivkins, Wortman Vaughan, Wu. COLT 2018, PMLR 75:1724–1738.
- Barman, Khan, Maiti, Sawarni. AAAI 2023, 37(6):6762–6769.
- Mansour, Slivkins, Syrgkanis. Operations Research 68(4):1132–1161, 2020.
- Drugan & Nowé. IJCNN 2013, pp. 1–8.
- Combes, Magureanu, Proutiere. NeurIPS 2017; pages 1763–1771 from one source only.
