# Layer 1 — LLM proposer

Layer 1 swaps the *proposer* for a language model while keeping **everything else
identical**: the same `Task` (data + grammar), the same `Verifier` (reward =
`1/(1+MSE)` − penalties), the same `Proposer.ask()/tell()` contract, and the same
metrics. That is the whole reason Layer 0 was built behind clean seams.

## Status

- **Arm 1 — program-database evolution: IMPLEMENTED.**
  [`llm_evolution.py`](llm_evolution.py) + [`../run_layer1.py`](../run_layer1.py).
  Runs an MLX model (default `Qwen2.5-3B-Instruct-4bit`) as the proposer; the LLM's
  free-form output is parsed into the SAME `Node` representation
  (`sia.expression.parse_expression`) and scored by the SAME `Verifier`.
- **Arms 2 & 3 — greedy / risk-seeking LoRA: IMPLEMENTED, NOT YET RUN ON-DEVICE.**
  [`lora_proposer.py`](lora_proposer.py). Same `ask()/tell()` seam; instead of an
  archive it attaches a **LoRA adapter** and takes one reward-weighted
  policy-gradient step per batch. The greedy vs. risk difference is *only* the
  per-sample weight `w_i`, imported from the shared
  [`sia/objectives.py`](../src/sia/objectives.py) — the SAME formulas Layer 0 uses.
  The portable loop logic (parsing, budget accounting, weighting) is unit-tested
  ([`../tests/test_layer1.py`](../tests/test_layer1.py),
  [`../tests/test_objectives.py`](../tests/test_objectives.py)), but the actual MLX
  LoRA training step has **not** been executed — the dev box has no Apple Silicon.
  See **On-device verification** below for the M4 smoke test the human must run.

```bash
pip install -e ".[layer1]"   # from the project root; mlx-lm (Apple Silicon only)
# arm 1 (evolution):
python ../run_layer1.py --arm evolution   --target medium --budget 64 --batch-size 8
# arms 2 & 3 (LoRA):
python ../run_layer1.py --arm greedy_lora --target medium --budget 64 --batch-size 8
python ../run_layer1.py --arm risk_lora   --target medium --budget 64 --batch-size 8
```

## On-device verification (M4 — REQUIRED; not run here)

The LoRA path touches MLX, which cannot run on this Linux dev box, so it is
**unverified end-to-end**. On an Apple-Silicon Mac, run this smoke test and check
the expected behavior before trusting the results:

```bash
pip install -e ".[layer1]"                        # from the project root; mlx-lm >= 0.31
# 1) sanity: the portable parts pass anywhere (no MLX needed)
PYTHONPATH=../src        python -m tests.test_objectives
PYTHONPATH=../src:..     python -m tests.test_layer1
# 2) tiny on-device smoke: one model, one target, budget 16 (2 LoRA steps @ batch 8)
python ../run_layer1.py --arm greedy_lora --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --target medium --budget 16 --batch-size 8 --max-tokens 48
python ../run_layer1.py --arm risk_lora   --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --target medium --budget 16 --batch-size 8 --max-tokens 48
# 3) full three-arm sweep config (start small, then scale seeds/budget):
python ../run_layer1.py --config ../configs/layer1_lora.yaml --quick --out ../results_layer1_lora
```

**Expected behavior if the MLX path is correct:**

- Both LoRA smokes complete without raising; per-batch lines print
  `calls=N/16 best=… top=…` exactly like the evolution arm.
- `diagnostics()["loss"]` is finite (printed indirectly via the run; you can add a
  print in `run_one` if you want to watch it). A reward-weighted PG loss can be
  positive or negative and need not decrease monotonically — what matters is that
  it is finite and that gradients flow (no NaNs).
- `valid_fraction` is in `[0, 1]`; early on it may be low (the base model rarely
  emits a clean in-grammar formula), and should not crash on invalid parses.
- Memory stays bounded (rank-8 LoRA on the last 8 blocks of a 3B-4bit is small).

**The one likely breakage point** is the `mlx_lm` API drift. If
`linear_to_lora_layers(model, num_layers, config)` raises, adapt the single call
in `LoRAProposer._setup_model` to your installed `mlx_lm` version (older builds
nest the config under `lora_parameters` or use `alpha` instead of `scale`). The
call site is isolated and commented for exactly this reason.

Once the smoke passes, the three-arm head-to-head (`configs/layer1_lora.yaml`)
answers the transfer question below. Do **not** launch the long sweep before the
smoke is green.

**Preliminary observation (smoke runs, 1 seed, Qwen2.5-3B-4bit, budget ≤ 64):** the
LLM evolution arm does *not* solve `medium` (`x²+sin(x)`) — unlike Layer 0, where
random/GP/risk all eventually do. With a generic prompt the 3B collapses onto a
linear `x+1`; nudged toward nonlinear terms it explores *products* like
`x*x*sin(x)` but not the *sum* `x²+sin(x)`. This is consistent with the transfer
question below (a target far from the model's prior is hard to reach) — but it is a
1-seed smoke, not a result. Real evaluation needs: more budget, multiple seeds,
bigger models (7B is cached), and prompt iteration, then a head-to-head of the three
arms on the same budget.

## The seam

Three shared pieces make all arms plug into Layer 0's machinery unchanged:

1. **Prompt** — built from the *same* task object (data points + grammar). The
   evolution arm appends best-so-far expressions as inspiration; the LoRA arms
   show the **data only** (they learn from rewards via weight updates, not
   in-context exemplars).
2. **Parse** — `sia.expression.parse_expression` turns the model's free-form text
   back into the *same* `Node` representation, so the *same* `Verifier` scores it.
   Invalid parse → an `INVALID` sentinel → reward 0 (the verifier already handles
   this), and the wasted generation still costs one verifier call (fair budget).
3. **`ask()/tell()`** — `ask()` samples a batch of completions; `tell()` either
   updates the archive (evolution) or takes one LoRA gradient step (greedy/risk).

Because the contract is unchanged, `run_layer1.run_one` drives every arm
identically, and `make_layer1` regenerates the Layer 1 figures/table.

The historical design skeleton ([`proposer_llm_stub.py`](proposer_llm_stub.py))
is kept only as a pointer to the real implementations
([`llm_evolution.py`](llm_evolution.py), [`lora_proposer.py`](lora_proposer.py)).

## The three arms (mirror Layer 0)

| Layer 0 arm | Layer 1 arm | Mechanism |
|---|---|---|
| Evolution (GP) | Program-database evolution ([`llm_evolution.py`](llm_evolution.py)) | [AlphaEvolve](https://arxiv.org/abs/2506.13131) / [ThetaEvolve](https://arxiv.org/abs/2511.23473) style: keep an archive of high-reward expressions, feed the best back into the prompt as inspiration. No weight updates. |
| Greedy RL (`E[R]`) | Greedy LoRA ([`lora_proposer.py`](lora_proposer.py)) | One LoRA gradient step per batch with `w_i = R_i − mean(R)` (`objectives.greedy_weights`). Expected-reward policy gradient / weighted SFT. |
| Risk-seeking / entropic RL | Risk-seeking LoRA ([`lora_proposer.py`](lora_proposer.py)) | Same LoRA step but `w_i` from the top-ε quantile (DSR, `objectives.quantile_weights`), the `e^{βR}`-tilt (`objectives.entropic_weights`), or — via `mode: cvar` — the risk-averse lower-tail mirror (`objectives.cvar_weights`). The weighting is the SAME source of truth as Layer 0's `rl_risk`. |

The greedy and risk LoRA arms differ in **exactly one** function — `_weights` —
which dispatches into `sia/objectives.py`. Everything else (prompt, sampling, the
weighted-PG loss, the Adam step) is identical between them.

## Tech

- Model: Qwen2.5-3B/7B-4bit via `mlx-lm` on the M4 MacBook (64GB) — same models
  the evolution arm already uses.
- LoRA via `mlx_lm.tuner.utils.linear_to_lora_layers`; the weighted-PG step is a
  hand-written `nn.value_and_grad` + `mlx.optimizers.Adam` update over the LoRA
  params only (base model frozen). See `LoRAProposer._lora_step`.
- Verifier-call budget stays the fair unit: 1 call per generation; LoRA gradient
  steps are free (they consume no verifier budget), exactly mirroring Layer 0.
- All `mlx`/`mlx_lm` imports are lazy, so this module imports fine on any box and
  Layer 0 stays numpy-only.

## The transfer question

Does the Layer 0 ranking — **evolution ≈ risk-seeking > greedy, with greedy
mode-collapsing** — survive an LLM proposer? Or does the pretrained prior change
the dynamics (e.g. prevent the greedy collapse, because the prior already spreads
mass over diverse, structured expressions)? That is the hypothesis Layer 1 tests,
and the reason Layer 0 establishes the non-LLM baseline first.
