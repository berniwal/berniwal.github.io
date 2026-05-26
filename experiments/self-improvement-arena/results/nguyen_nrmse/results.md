# Nguyen 1–8 — recovery under NRMSE reward

Standard Nguyen suite, DSR-faithful setup: `L₀ = {+, −, ×, ÷, sin, cos, exp, log, x}` (no
constant terminals; constants are constructed, e.g. `1 = x/x`), per-benchmark sampling
ranges, 25 seeds, **NRMSE reward** (the MSE reward `1/(1+MSE)` saturates and suppresses
exact recovery — see the post-fix note in `results/layer0_*/results.md`), 2M verifier calls
per run. Symbolic recovery = exact SymPy equivalence with const-snap, **recomputed locally**
from the per-run `best_expr` so it does not depend on the pod having sympy installed.

Run: `nguyen-nrmse-2026-05-25` (RunPod CPU, ~1h), 800 logs.

## Symbolic recovery / 25 seeds

| target | formula | Random | Evolution (GP) | Greedy (= VPG) | Risk-seeking |
|---|---|---:|---:|---:|---:|
| nguyen-1 | x³ + x² + x | **25** | 22 | 24 | **25** |
| nguyen-2 | x⁴ + x³ + x² + x | 22 | 10 | 24 | **25** |
| nguyen-3 | x⁵ + … + x | 0 | 4 | 22 | **23** |
| nguyen-4 | x⁶ + … + x | 0 | 1 | 19 | **22** |
| nguyen-5 | sin(x²)·cos(x) − 1 | 0 | 3 | 0 | **11** |
| nguyen-6 | sin(x) + sin(x + x²) | 0 | 15 | **25** | **25** |
| nguyen-7 | log(x+1) + log(x²+1) | 0 | 0 | 0 | 0 |
| nguyen-8 | sqrt(x) = exp(x/(x+x)·log x) | 0 | 1 | 0 | **16** |
| **average** | **/200** | **47 (24%)** | **56 (28%)** | **114 (57%)** | **147 (74%)** |

## Observations

- **Risk-seeking wins on average** (**74% vs greedy 57%**), is the **only method to recover
  Nguyen-5** (`sin(x²)cos(x) − 1`) — the rare-outlier regime the risk-seeking objective is
  designed for — and is the only method to crack **Nguyen-8** at scale (16/25 ≈ 64%, vs
  greedy 0/25).
- **Greedy (VPG) does not collapse on Nguyen-3/4** here (88%, 76%) — the DSR paper's 4%, 1%
  rates are an artifact of the 2021 config (shorter max length, no soft-length / uniform-arity
  priors), confirmed by re-running DSR's own current code (8/8 on Nguyen-3).
- **Nguyen-8 is mostly solved by risk-seeking (16/25 = 64%); Nguyen-7 isn't solved by anyone.**
  Both are expressible in `L₀` — DSR notes the Nguyen-8 construction as `exp(x/(x+x) · log(x))`
  and risk-seeking finds it on 16 of 25 seeds. DSR reports ~96% on Nguyen-8 with the same
  grammar; the remaining 64→96 gap points to implementation details we have not rigorously
  matched — DSR's exact training-loop schedule (entropy / lr decay over the 2M steps), per-step
  sampling logic, and other code-level specifics. We share the obvious machinery: same grammar, same hard length cap, same invalid
  handling (reward 0 for NaN / overflow). Nguyen-7 needs the +1 constant constructed *inside*
  two separate `log` arguments and isn't found by any of the four — investigating which
  implementation detail closes it is the open follow-up.
- **A measurement bug in the earlier draft.** An earlier version of this table read all-zeros
  for Nguyen-8; the policy was finding the right answer but our offline parser rejected any
  string containing `exp`/`log`, so the symbolic check returned False. Fixed (`_AST_FUNCS`
  now includes the Koza unary library) — the policy was fine all along.
- The earlier sweep at MSE reward gave **misleading numbers** (greedy avg 0.39, risk 0.19 —
  inverted) because the MSE reward saturates and the policy plateaus on a numerically-close
  but wrong expression. The nrmse reward keeps a usable gradient toward exact recovery.

Compare to `runpod/run_dsr.sh` (the arena's cross-check against DSR's own released code) and
`results/layer0_nrmse/results.md` (our three synthetic targets).
