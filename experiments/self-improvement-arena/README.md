# Self-Improvement Arena

A small, controlled testbed for the optimisation dynamic at the heart of methods like
[Deep Symbolic Regression](https://arxiv.org/abs/1912.04871), [AlphaEvolve](https://arxiv.org/abs/2506.13131),
and [TTT-Discover](https://arxiv.org/abs/2601.16175): four search methods compete on the
same task, scored by the same verifier, under the same budget; only the proposer changes.
The task is symbolic regression (recover `f(x)` from `(x, y)` points) because it is fast,
honest about success (exact symbolic recovery is verifiable), and works in seconds on a
laptop.

📝 **Full narrative + interactive widgets in the blog post:**
[Watching Search Algorithms Discover a Formula](https://berniwal.github.io/#/blog/visualizing-symbolic-regression).
This README is the operating reference — install, run, where the numbers live.

## What's in here

```
src/sia/                 the framework, all numpy
  expression.py            grammar, tree eval, sympy check, GP operators
  task.py                  benchmark targets + Nguyen suite + data generation
  verifier.py              fixed reward + success check + call counter
  policy.py                numpy RNN/LSTM policy + manual BPTT (shared by RL arms)
  objectives.py            shared advantage formulas (greedy/quantile/entropic/cvar)
  proposers/               random, gp, greedy, risk -- the swappable part
  runner.py                fair-budget ask/tell loop, multi-seed, resumable
  metrics.py, plotting.py  curves + diversity + results.md
layer1/                  LLM proposers behind the SAME ask/tell seam
  torch_lora_proposer.py   GPU LoRA + GRPO (Qwen2.5-0.5B); used in the blog
  lora_proposer.py         Apple-Silicon MLX equivalent (kept; off by default)
  llm_evolution.py         frozen-LLM, AlphaEvolve-style in-context evolution
configs/                 one YAML per sweep (single source of HPs)
  layer0.yaml              100k smoke for the four Layer-0 methods
  scaling.yaml             canonical 2M Layer-0 sweep, MSE reward
  scaling_nrmse.yaml       canonical 2M Layer-0 sweep, NRMSE reward
  nguyen.yaml              Nguyen 1-8 sweep (MSE; saturates -- see results.md)
  nguyen_nrmse.yaml        Nguyen 1-8 sweep (NRMSE; the headline)
  layer1.yaml              MLX in-context evolution arm
  layer1_lora.yaml         MLX LoRA arms
run_layer0.py            single-process numpy driver for the four Layer-0 arms
run_layer1.py            MLX LoRA + in-context evolution driver
run_layer1_torch.py      torch LoRA + GRPO driver (used for the blog's LLM runs)
export_replay.py         single-seed step-through bake -> results/layer0_*/replay.json
export_blog_data.py      bundles replays + LLM summaries into public/data/ for the blog
app_engine.py            steppable engine behind the Streamlit visualiser
streamlit_app.py         the interactive visualiser
tests/                   grammar round-trips, advantage formulas, ask/tell shape, etc.
results/                 committed artifacts: results.md + replay.json per sweep
                         (per-run logs/ are gitignored)
```

The cloud harness for the larger sweeps (RunPod + GCS-backed result sync, used to produce
the LLM arms in the blog) lives one level up at [`experiments/runpod/`](../runpod/).

## Findings (cross-linked to the blog)

1. **Risk-seeking RL recovers `harder` where greedy never does.** On our 2M-call NRMSE
   sweep (20 seeds), risk-seeking solves `harder = x³ − x + cos(2x)` 75% of the time;
   evolution 15%; greedy 0%. *Same policy class, same gradient machinery — only the
   per-sample advantage formula `wᵢ` changes.*

2. **The mechanism is an entropy *rebound*.** Per-step policy entropy on the 2M replay
   shows greedy and entropic both *committing* to wrong attractors around ~300k calls
   (H drops to ≈0.5 / ≈0.1 and stays there). Risk-seeking commits too — and then
   *climbs back up* to H ≈ 1.8, re-explores, and finds the cosine term. The hard top-ε
   quantile is the only arm whose entropy rebounds, which is, on this target, the
   difference between recovery and a permanent ceiling.

3. **The ranking transfers to an LLM proposer.** Swap the numpy RNN for Qwen2.5-0.5B
   trained with LoRA + GRPO under the *same* advantage formulas (`sia.objectives` is
   imported by both proposers). On `harder` the order matches Layer 0: risk-seeking 5/5
   numeric > best-of-N 3/5 > entropic 2/5 > greedy / evolution 1/5. *Symbolic* recovery
   drops though (the 150 k-token vocabulary makes the exact closed form much harder to
   hit than the 10-token grammar).

4. **On Nguyen 1-8 (DSR's official benchmark) risk-seeking wins on average and uniquely
   cracks Nguyen-5.** Recovery (25 seeds, 2M, NRMSE, recomputed locally with SymPy):
   random 24% / GP 28% / greedy 57% / risk **74%**. Nguyen-5 (`sin(x²)·cos(x) − 1`) is
   solved by risk-seeking 11/25 and greedy 0/25 — the rare-outlier regime the
   risk-seeking objective is designed for.

5. **Caveats.** Nguyen-7 (`log(x+1) + log(x²+1)`) is unsolved by any of our arms at 2M
   (DSR reports ~35%); a 27-combo `(lr, ent_coef, ε)` grid search on it returned 0/5
   across every cell, so the gap is *not* in static hyperparameters — it lives in
   DSR's training-loop schedule (entropy / lr decay over the 2M steps) or other
   code-level specifics we have not rigorously matched. See the blog post.

## Quick start

```bash
# Layer 0 only (numpy, runs anywhere)
pip install -e .
python -m pytest -q                                         # 49 tests
python run_layer0.py --quick                                # fast smoke -> results_quick/

# the canonical 2M Layer-0 sweeps (parallel; ~1-1.5h each on an M4)
python run_layer0.py --config configs/scaling_nrmse.yaml --out results/layer0_nrmse --parallel
python run_layer0.py --config configs/scaling.yaml       --out results/layer0_mse   --parallel

# the Nguyen 1-8 sweep that produced finding (4)
python run_layer0.py --config configs/nguyen_nrmse.yaml --out results/nguyen_nrmse --parallel

# the LLM Layer-1 run (Qwen2.5-0.5B + LoRA + GRPO; GPU)
pip install -e ".[layer1-gpu]"                              # adds torch + transformers + peft
python run_layer1_torch.py --target harder --arm risk --mode quantile --rounds 80 --batch 32
```

```bash
# interactive visualiser (no sweep, step the search batch-by-batch)
pip install -e ".[app]"
streamlit run streamlit_app.py
```

For the cloud (RunPod) versions of the LLM sweep and the Nguyen NRMSE sweep, see
[`runpod/run_layer1_torch.sh`](runpod/run_layer1_torch.sh) and
[`runpod/run_dsr.sh`](runpod/run_dsr.sh) (the cross-check against DSR's own code) —
those are the arena-specific pod scripts. The shared harness (launch / fetch /
GCS sync) lives one level up in [`../runpod/`](../runpod/).

## Where the numbers live

- [`results/layer0_nrmse/results.md`](results/layer0_nrmse/results.md) — the headline
  Layer-0 table (20 seeds × 2M × NRMSE).
- [`results/layer0_mse/results.md`](results/layer0_mse/results.md) — same under MSE
  (the saturating reward; risk-seeking only cracks `harder` 10% here, vs 75% on NRMSE).
- [`results/nguyen_nrmse/results.md`](results/nguyen_nrmse/results.md) — the Nguyen
  1-8 table + the measurement-bug note explaining why an earlier draft reported
  Nguyen-8 as 0/25.
- [`results/layer1-torch/`](results/layer1-torch/) — per-(arm, target, seed)
  `summary.json` files for the LLM transfer (pulled from RunPod via the cloud harness).

The `replay.json` files under `results/layer0_*` power the blog's "Watch them evolve"
widget (one representative seed, 2M log-spaced frames).

## Honesty / caveats

- The numbers reported here use exact SymPy equivalence (with a small `const_tol` snap
  for fitted constants) — DSR's strictest definition of recovery.
- Numbers come from the post-fix run. An earlier draft of this work had a normalisation
  bug in the risk-seeking gradient (missing the 1/(εN) scale) that suppressed
  risk-seeking's signal; that's fixed in [`src/sia/policy.py`](src/sia/policy.py).
- On Nguyen-3 / Nguyen-4 our vanilla PG (= DSR's VPG) recovers 88% / 76% — the DSR
  paper reports 4% / 1%. We cross-checked DSR's currently-released code (see
  [`runpod/run_dsr.sh`](runpod/run_dsr.sh)): their VPG also recovers Nguyen-3 ≈100%,
  so the paper's 4% / 1% is a 2021-config artifact, not a property of vanilla PG. The
  qualitative ranking (risk-seeking wins on hard targets, uniquely cracks Nguyen-5)
  is unaffected.

## References

- DSR — Petersen et al. 2021, [arXiv:1912.04871](https://arxiv.org/abs/1912.04871);
  reference code at [dso-org/deep-symbolic-optimization](https://github.com/dso-org/deep-symbolic-optimization).
- AlphaEvolve — Novikov et al. 2025, [arXiv:2506.13131](https://arxiv.org/abs/2506.13131).
- TTT-Discover / entropic *J*_β — Jiang et al. 2025, [arXiv:2601.16175](https://arxiv.org/abs/2601.16175).
- RS-GRPO — Jiang et al. 2025, [arXiv:2509.24261](https://arxiv.org/abs/2509.24261).
- RiskPO — Ren et al. 2025, [arXiv:2510.00911](https://arxiv.org/abs/2510.00911).
- Howard & Matheson 1972 — *Risk-Sensitive Markov Decision Processes*,
  [DOI: 10.1287/mnsc.18.7.356](https://doi.org/10.1287/mnsc.18.7.356).
- Regularized Evolution — Real et al. 2019, [arXiv:1802.01548](https://arxiv.org/abs/1802.01548).

Co-authored with [Claude](https://www.anthropic.com/claude) (Anthropic) — code,
experiments, and interactive scaffolding by Claude, refined and co-designed by
Bernhard.
