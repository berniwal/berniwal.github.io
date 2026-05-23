# Self-Improvement Arena — Layer 0

A small, fast, **non-LLM** sandbox for studying one optimization dynamic from the
self-improvement / open-ended-search literature:

> **Evolution beats greedy RL; greedy RL mode-collapses to a simple-but-suboptimal
> solution; risk-seeking RL recovers the peak.**

Methods from that literature (AlphaEvolve, ThetaEvolve, TTT-Discover, Deep Symbolic
Regression) are expensive and LLM-bound, which makes the *underlying* search
dynamics hard to study. Here the same dynamic is reproduced on **symbolic
regression** — recovering a hidden formula from data — which runs in
seconds-to-minutes on a laptop CPU. Layer 1 (designed but not built; see
[`layer1/`](layer1/)) will swap in an LLM proposer behind the *same* interfaces to
test whether the conclusions transfer.

## The research question

Three method families compete on the **same task**, with the **same verifier**,
under the **same budget** (counted in *verifier calls*, so the comparison is fair).
Only the *proposer* changes:

1. **Evolution** — genetic programming (population-based search).
2. **Greedy RL** — a policy trained to maximize **expected** reward (standard
   REINFORCE). *Expected to collapse.*
3. **Risk-seeking / entropic RL** — the **same** policy trained to maximize the
   **best** outcomes, not the average. *Expected to recover the peak.*

## The setup (the seams)

Everything is built around three interfaces so that Layer 0 (search/RL) and Layer 1
(LLM) can share the task, the reward, and the metrics — only the proposer differs.

- **Task** ([`task.py`](src/sia/task.py)) — a hidden target function sampled into
  `(x, y)` data plus a held-out set, and the expression grammar (variables `{x}`,
  constants `{1, 2, 0.5}`, unary `{sin, cos}`, binary `{+, −, ×, ÷}`). Three targets
  of increasing difficulty, all reachable from the grammar:
  - `easy`: `x² + 1`
  - `medium`: `x² + sin(x)`  ← the main running example
  - `harder`: `x³ − x + cos(2x)`
- **Verifier** ([`verifier.py`](src/sia/verifier.py)) — the single fixed reward:
  `reward = 1/(1 + MSE_train) − length_penalty · complexity`; invalid expressions
  (div-by-zero, NaN, overflow) score 0. Microsecond-fast and it **counts its own
  calls**. Success = held-out MSE `< 1e-6` (recovering the function, not fitting the
  training x's).
- **Proposer** ([`proposers/base.py`](src/sia/proposers/base.py)) — the only thing
  that changes between methods. A simple `ask()` (propose a batch) / `tell()` (learn
  from rewards) contract. The [`runner`](src/sia/runner.py) drives the loop and
  stops at the shared budget.

## The three RL objectives — one gradient, three weightings

The whole reason the RL arms are written by hand in numpy
([`policy.py`](src/sia/policy.py)) is to make this point concrete. A policy-gradient
update is **always**

```
maximize   Σ_i  w_i · Σ_t log π(a_t^i)        (+ an entropy bonus)
```

and **the three arms differ only in the per-trajectory weight `w_i`** — pure scalar
arithmetic on the rewards. Backprop is identical.

| arm | objective | weight `w_i` | file |
|---|---|---|---|
| **Greedy** | `E[R]` | `R_i − mean(R)` | [`rl_greedy.py`](src/sia/proposers/rl_greedy.py) |
| **Risk-seeking (DSR)** | `(1−ε)` reward quantile | `(R_i − R̃_ε)` for top-ε, else `0` | [`rl_risk.py`](src/sia/proposers/rl_risk.py) |
| **Entropic (`J_β`)** | `(1/β) log E[e^{βR}]` | `∝ e^{βR_i}` (centered) | [`rl_risk.py`](src/sia/proposers/rl_risk.py) |

Optimizing the *average* sample (greedy) is the wrong target for discovery: you do
not want a typically-decent expression, you want the *one* exact hit. That pressure
is what collapses the greedy policy onto a simple, safe, wrong attractor.

## Results

<!-- RESULTS:BEGIN (regenerate with `python run_layer0.py`) -->
20 seeds per cell, budget = 100,000 verifier calls, `configs/layer0.yaml`.

| target | method | success rate | median evals-to-solve | mean best reward |
|---|---|---|---|---|
| easy | Random search | 1.00 | 9300 | 0.9944 |
| easy | Evolution (GP) | 1.00 | 1700 | 0.9945 |
| easy | Greedy RL | 0.75 | 7400 | 0.9134 |
| easy | Risk-seeking RL (DSR) | 1.00 | 5500 | 0.9950 |
| easy | Entropic RL (J_beta) | 0.85 | 3000 | 0.9669 |
| medium | Random search | 0.50 | 37000 | 0.9290 |
| medium | Evolution (GP) | **0.85** | 7200 | **0.9890** |
| medium | Greedy RL | **0.25** | 24000 | **0.8242** |
| medium | Risk-seeking RL (DSR) | 0.55 | 19400 | 0.9675 |
| medium | Entropic RL (J_beta) | 0.45 | 3200 | 0.8386 |
| harder | Random search | 0.00 | - | 0.5728 |
| harder | Evolution (GP) | 0.00 | - | 0.6520 |
| harder | Greedy RL | 0.00 | - | 0.3659 |
| harder | Risk-seeking RL (DSR) | 0.00 | - | 0.4718 |
| harder | Entropic RL (J_beta) | 0.00 | - | 0.6735 |

**The expected dynamic reproduces.** On the main example (`medium`, `x² + sin(x)`):

- **Evolution beats greedy RL** — GP 0.85 success / 0.989 best vs. greedy 0.25 / 0.824.
- **Greedy RL mode-collapses below random search** — greedy (0.25, 0.824) is *worse
  than just sampling randomly* (0.50, 0.929). The diversity plot shows why: greedy's
  batch diversity crashes toward zero within a few thousand calls and never recovers
  — it commits to one simple, wrong attractor. Greedy is the **worst method on every
  target** (best reward 0.913 / 0.824 / 0.366 on easy / medium / harder).
- **Risk-seeking RL recovers** — the DSR quantile arm (0.55, 0.968) climbs back
  above greedy *and* above random, and its diversity stays high throughout. It does
  not quite match GP here, but it cleanly undoes the collapse.

**Two honest wrinkles** (the point is the real result, not a clean story):

1. **Random search is a strong baseline** on this tiny grammar — on `medium` it ties
   the RL arms on success rate. The load-bearing signal is the *ordering* (greedy
   collapsing *below* random; DSR risk-seeking and evolution recovering *above* it),
   not the absolute numbers.
2. **The entropic arm (Jiang/TTT-style) behaves more like greedy here than like the
   DSR quantile.** With `β = 2` it collapses its diversity fast (see the diversity
   plot) and underperforms the hard top-ε quantile on `easy`/`medium` (0.85/0.45 vs.
   1.00/0.55), though it edges everything on `harder`. So in *this* setting the
   **hard top-ε quantile is the more robust risk-seeking variant**, and the soft
   exponential tilt needs its `β` tuned to avoid drifting back toward greedy
   behaviour. That is a useful, concrete illustration of the lineage point below.
<!-- RESULTS:END -->

![best-so-far reward vs. budget](results/curves.png)
![success rate](results/success.png)
![diversity (mode collapse)](results/diversity.png)

## The DSR ↔ Jiang lineage note

This sandbox's risk-seeking arm **is** essentially Deep Symbolic Regression
(Petersen et al., 2021): an RNN emits expression tokens and is trained with a
**risk-seeking policy gradient** on the top-ε reward quantile. That paper made the
"optimize the best, not the average" point for symbolic regression in 2021.

TTT-Discover (2026) uses an **entropic objective** `J_β = (1/β) log E[e^{βR}]` and
credits it to **Jiang et al. (2025)** — *not* to DSR. Yet the entropic objective is
just the **soft** version of the same idea: `β → ∞` recovers the max (pure
risk-seeking), and DSR's hard top-ε quantile is essentially a limiting case of the
soft exponential tilt. Both abandon expected reward for the same reason. The
2021 → 2025 gap with no apparent cross-citation is a small but telling example of
how the same optimization insight gets rediscovered across sub-communities (RL for
program search vs. test-time LLM training). Both arms are included here
(`risk` = DSR quantile, `risk_entropic` = Jiang/TTT) so the two can be compared
directly.

## How to run

```bash
pip install -r requirements.txt
python run_layer0.py --quick                         # fast smoke run -> results_quick/
python run_layer0.py --config configs/layer0.yaml    # full headline run (20 seeds) -> results/
PYTHONPATH=src python -m tests.test_core             # sanity checks
```

Everything is seeded and configured from [`configs/layer0.yaml`](configs/layer0.yaml)
(budget, seeds, all hyperparameters). Outputs land in [`results/`](results/):
`curves.png`, `success.png`, `diversity.png`, `table.md`, and raw per-run JSON logs.

## Repo layout

```
src/sia/
  expression.py      grammar, tree eval, complexity, prefix tokens, GP operators
  task.py            benchmark targets + data generation
  verifier.py        the fixed reward + success check + call counting
  policy.py          numpy vanilla-RNN token policy + manual BPTT (shared by RL arms)
  proposers/         random, gp, greedy, risk  (the pluggable part)
  runner.py          fair-budget ask/tell loop, multi-seed
  metrics.py         best-so-far, success rate, diversity, cross-seed aggregation
  plotting.py        the three figures + the results table
configs/layer0.yaml  single source of truth
run_layer0.py        one command -> all figures + table
tests/test_core.py   grammar round-trips, exact-target reward, call counting
layer1/              LLM-proposer design note + interface stub (NOT built in v1)
results/             generated figures + table (committed); raw logs (gitignored)
```

## Honesty / caveats

- Results are reported over 20 seeds with error bands; a single lucky run proves
  nothing. If the expected dynamic had failed to reproduce, that would be reported
  here rather than tuned away — the honest result is the point.
- Random search is a **strong** baseline on this tiny grammar; the meaningful
  signal is the *ordering* (greedy collapsing *below* random, risk-seeking and
  evolution recovering above it), not the absolute rewards.
- `harder` (`x³ − x + cos(2x)`) is genuinely hard for exact recovery at this budget;
  it is reported as a best-reward comparison, not a solved task.
- The budget is enforced as "stop once `verifier.calls ≥ budget`," so methods may
  overshoot by at most one batch — negligible and equal across methods.

## References

- **Deep Symbolic Regression** — Petersen et al., ICLR 2021. Risk-seeking policy
  gradient for symbolic regression. [arXiv:1912.04871](https://arxiv.org/abs/1912.04871)
- **Regularized Evolution for Image Classifier Architecture Search** — Real et al.,
  AAAI 2019. Evolution competitive with / faster than RL on a verifiable reward.
  [arXiv:1802.01548](https://arxiv.org/abs/1802.01548)
- **TTT-Discover** — 2026. Entropic objective `J_β` with adaptive `β` (credits Jiang
  et al. 2025 for the objective). [arXiv:2601.16175](https://arxiv.org/abs/2601.16175)
- **ThetaEvolve** — 2025. LLM program-database evolution; relevant for Layer 1.
  [arXiv:2511.23473](https://arxiv.org/abs/2511.23473)
```
