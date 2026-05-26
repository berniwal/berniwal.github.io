# Experiment: Does reasoning help the Layer-1 LLM proposer?

Status: scaffolding merged, runs **not yet executed**. The agent reading this should
execute the runbook in [§ Runbook](#runbook), then write up results in
[§ Reporting](#reporting).

## Background

The Layer-1 LLM arm currently disallows reasoning. In
`layer1/torch_lora_proposer.py` the prompt instructs the model to "Reply with ONLY
the formula for f(x) on a single line — no words, no 'y =', no code fences", and the
sampler caps generation at `max_new_tokens=48`. The model in use is
`Qwen/Qwen2.5-0.5B-Instruct`, which predates the dual-mode reasoning generation.

TTT-Discover (Stanford / NVIDIA / Together, [arxiv:2601.16175](https://arxiv.org/abs/2601.16175),
[github](https://github.com/test-time-training/discover)) treats reasoning as the
core of the rollout: their two-phase sampler gives the model up to **26 000 tokens
of thinking** per rollout before forcing a final-answer phase, and they train
through the reasoning tokens (the thinking-phase mask is `1.0`, not `0.0`). So our
current setup matches TTT-Discover on the per-task-adaptation axis (fresh LoRA per
target, no carry-over) but is the *opposite* of TTT-Discover on the
reasoning-length axis.

## Hypothesis

**H1 — Reasoning unlocks structural hypothesis-formation.** On `harder`
(`x^3 − x + cos(2x)`), the Layer-0 RNN recovers the formula exactly because its
10-token grammar makes the search space small; the Layer-1 LLM samples from a
~150k-token vocabulary and almost never recovers the exact skeleton (see the
"Does the ranking transfer to an LLM?" section of the blog post). The bottleneck
is identifying the right *structural template* — "polynomial + periodic" — not
fitting tokens to it. If allowed to think, a reasoning-capable model should be
able to name the template explicitly and then emit it, lifting exact symbolic
recovery rate on `harder`.

**H2 — Risk-seeking still wins.** Allowing reasoning changes the rollout content
but not the per-sample advantage formula. The same Layer-0 ranking should hold:
the **risk** arm should beat **greedy** by a wider margin once reasoning is on,
because risk-seeking is the arm that rewards rare excellent outliers — and
"correct skeleton" is exactly such an outlier in the LLM's output distribution.

**H3 — RL trains the reasoning trace, not just the final answer.** The GRPO update
in `torch_lora_proposer.py` already flows gradient through *all* generated tokens
(both `<think>...</think>` and the final formula). Inspecting the reasoning traces
across rounds should show a measurable shift: from generic "let me check the
signs" patter early on, to more diagnostic structural language ("looks cubic with
a periodic correction") late in training. This is the actual "self-improvement"
claim — does the policy learn to *search* better, not just to fit one target.

## What changed in the code

- `layer1/torch_lora_proposer.py`
  - New constructor flag `reasoning: bool = False`.
  - When on, the prompt drops the "ONLY the formula, one line" instruction and
    asks the model to think step-by-step inside a thinking block, then emit the
    formula as the **last** line. The chat template is invoked with
    `enable_thinking=True` (Qwen3-style); the call falls back gracefully for
    tokenizers that don't accept the kwarg.
  - New static helper `_extract_formula(text)` strips everything up to and
    including the last `</think>` marker (if present) and returns the last
    non-empty line. Non-reasoning behaviour is unchanged — first-line and
    last-line of a single-line reply are the same.
  - A console warning fires when `reasoning=True` but `max_new_tokens<256`.
- `run_layer1_torch.py` — new `--reasoning` flag, threaded through.
- `runpod/run_layer1_torch.sh` — new `REASONING=1` env knob; when set, appends
  `--reasoning` and bumps `MAXNEW` default from `48` → `2048`.

No existing default behaviour changes — every flag is opt-in.

## Runbook

Run these on a GPU pod with at least 24 GB VRAM (Qwen3-1.7B + LoRA + batch=16 fits
comfortably on a single A10/L4; an A100/H100 is overkill but faster). All commands
assume `cd experiments/self-improvement-arena` and that the torch + transformers +
peft + scipy extras are installed (see `runpod/run_layer1_torch.sh` for the
exact pin).

### Step 1 — sanity check that nothing in the non-reasoning path regressed

```bash
python3 run_layer1_torch.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --target medium --arm risk --mode quantile \
    --rounds 10 --batch 8 --max-new-tokens 48 \
    --seed 0 --out results/sanity-noreasoning
```

Expected: identical reward trajectory to the pre-PR runs (within sampling noise
of the same seed); `valid` should stay near pre-PR levels.

### Step 2 — headline A/B at one seed

Pick `harder` as the target because it is the only one where Layer 1 has a
non-trivial ceiling. One seed first to verify the harness works end-to-end before
spending money on a sweep.

```bash
# Treatment: reasoning ON
python3 run_layer1_torch.py \
    --model Qwen/Qwen3-1.7B --reasoning \
    --target harder --arm risk --mode quantile \
    --rounds 40 --batch 16 --max-new-tokens 2048 \
    --lr 1e-6 --temperature 1.0 \
    --seed 0 --out results/reasoning-on

# Control: reasoning OFF (same model, same everything)
python3 run_layer1_torch.py \
    --model Qwen/Qwen3-1.7B \
    --target harder --arm risk --mode quantile \
    --rounds 40 --batch 16 --max-new-tokens 64 \
    --lr 1e-6 --temperature 1.0 \
    --seed 0 --out results/reasoning-off
```

Note both arms use **Qwen3-1.7B**, not Qwen2.5-0.5B, so the only difference is
`enable_thinking` plus the prompt change. That isolates "reasoning helps" from
"a bigger / newer model helps".

### Step 3 — multi-seed sweep, two arms

Once Step 2 succeeds, sweep `risk` vs `greedy` × reasoning on/off across 5 seeds.
Easiest from the RunPod harness:

```bash
# In runpod/run_layer1_torch.sh callers (one per cell):
TARGET=harder ARM=risk   MODE=quantile MODEL=Qwen/Qwen3-1.7B REASONING=1 SEEDS=0,1,2,3,4 ./runpod/run_layer1_torch.sh
TARGET=harder ARM=risk   MODE=quantile MODEL=Qwen/Qwen3-1.7B REASONING=0 MAXNEW=64 SEEDS=0,1,2,3,4 ./runpod/run_layer1_torch.sh
TARGET=harder ARM=greedy                MODEL=Qwen/Qwen3-1.7B REASONING=1 SEEDS=0,1,2,3,4 ./runpod/run_layer1_torch.sh
TARGET=harder ARM=greedy                MODEL=Qwen/Qwen3-1.7B REASONING=0 MAXNEW=64 SEEDS=0,1,2,3,4 ./runpod/run_layer1_torch.sh
```

This is a 2×2×5 = 20-run grid. At ~5–10 min per run on an L4 (rough estimate;
verify on Step 2), that is 1.5–3 GPU-hours. Stop early if the Step-2 result is
unambiguous.

### Step 4 — inspect reasoning traces

The proposer already exposes `last_io()` returning `{prompt, responses}`. Add a
short script that, for one of the `reasoning-on` runs, dumps the last batch's
responses to a file:

```python
# inspect_traces.py — run after a training run finishes
import json, sys
from layer1.torch_lora_proposer import TorchLoRAProposer  # reuse the class to re-sample
# OR: easier — log prop.last_io() each round inside run_layer1_torch.py and inspect the JSON.
```

(Cheaper and more useful: add `last_io=prop.last_io()` into the per-round history
dict in `run_layer1_torch.py:90` so the traces are persisted in `summary.json`.
Do this before Step 3 if disk allows.)

## Reporting

When the runs finish, write a brief results summary (1–2 paragraphs + a small
table) into `docs/REASONING_RESULTS.md` covering:

1. **Best-so-far reward at round 40** on `harder`, mean ± std across 5 seeds, for
   each of the 4 cells (arm × reasoning). Headline: does reasoning-on beat
   reasoning-off at the same model?
2. **Exact symbolic recovery rate** (`symbolic_solved_at != None`) across the 5
   seeds, same cells. This is the H1 metric.
3. **Sample efficiency** — at what round does reasoning-on first hit
   reward ≥ 0.9? Compare to reasoning-off.
4. **Trace inspection** — pick 3 reasoning traces from round 1 and 3 from round 40
   of the best `risk + reasoning` seed. Paste them inline. Does the language
   shift toward structural hypothesis-formation? (H3.)
5. **Honest negative results** — if reasoning *doesn't* help on `harder`, that is
   also publishable. Two likely explanations to discuss: (a) the per-round budget
   doubled from ~768 tokens (16 × 48) to ~32 768 (16 × 2048), so the comparison
   may need a wall-clock or token-budget axis instead of rounds; (b) at 1.7B
   params the math prior is still too thin for reasoning to help. If (a) looks
   live, re-run reasoning-off with `--batch 64 --max-new-tokens 64` to roughly
   token-match.

## Open questions for follow-up

- **Larger model**: Qwen3-8B-thinking should sharpen H1; not worth the GPU budget
  until the 1.7B result is in.
- **Token-budget honesty**: the verifier-call axis in the blog post is the right
  one for Layer 0; for Layer 1 with reasoning, total tokens generated is a more
  meaningful budget. Worth adding a parallel x-axis in any chart.
- **TTT-Discover-faithful comparison**: their two-phase sampler is not yet
  reproduced here. If reasoning helps even with the naive single-phase setup,
  that strengthens the case for the two-phase upgrade as a next step.
