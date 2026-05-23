# Layer 1 — LLM proposer (design only; not built in v1)

Layer 1 swaps the *proposer* for a language model while keeping **everything else
identical**: the same `Task` (data + grammar), the same `Verifier` (reward =
`1/(1+MSE)` − penalties), the same `Proposer.ask()/tell()` contract, and the same
`runner`/metrics/plots. That is the whole reason Layer 0 was built behind clean
seams — Layer 1 should be a drop-in `LLMProposer`.

## The seam

`proposer_llm_stub.py` is a skeleton with three TODO points:

1. `format_prompt(task, inspirations)` — turn the *same* task object into a prompt
   (data points + grammar; plus best-so-far expressions for the evolution arm).
2. `parse_expression(text)` — turn the model's text back into the *same* `Node`
   representation, so the *same* `Verifier` scores it. Invalid parse → `None` →
   reward 0 (the verifier already handles this).
3. `LLMProposer.ask()/tell()` — sample completions; update the archive (evolution)
   or take a LoRA step (greedy / risk).

Because the contract is unchanged, `runner.run_method` drives the LLM proposer
with no modification, and the Layer 0 figures/table regenerate for Layer 1.

## The three arms (mirror Layer 0)

| Layer 0 arm | Layer 1 arm | Mechanism |
|---|---|---|
| Evolution (GP) | Program-database evolution | AlphaEvolve / ThetaEvolve style: keep an archive of high-reward expressions, feed the best back into the prompt as inspiration. No weight updates. |
| Greedy RL (`E[R]`) | Greedy LoRA | Fine-tune a LoRA adapter on high-reward outputs (SFT / expected-reward policy gradient). |
| Risk-seeking / entropic RL | Risk-seeking LoRA | LoRA step on the top-ε outputs (DSR) or the `e^{βR}`-weighted outputs (entropic). Reuse Layer 0's `rl_risk` weighting verbatim. |

## Tech (when built)

- Model: Qwen2.5-0.5B / 1.5B via `mlx-lm` on the M4 MacBook (64GB).
- LoRA fine-tuning via `mlx-lm`'s LoRA utilities for the greedy/risk arms.
- Keep the verifier-call budget as the fair unit, exactly as in Layer 0.

## The transfer question

Does the Layer 0 ranking — **evolution ≈ risk-seeking > greedy, with greedy
mode-collapsing** — survive an LLM proposer? Or does the pretrained prior change
the dynamics (e.g. prevent the greedy collapse, because the prior already spreads
mass over diverse, structured expressions)? That is the hypothesis Layer 1 tests,
and the reason Layer 0 establishes the non-LLM baseline first.
