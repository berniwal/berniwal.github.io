"""HISTORICAL DESIGN NOTE — superseded by the real implementations.

This file was the original Layer 1 skeleton (design only). All three arms are now
implemented; this module is kept only as a signpost so links to it still resolve.

    Arm 1  evolution    -> layer1/llm_evolution.py   (LLMEvolutionProposer)
    Arm 2  greedy LoRA  -> layer1/lora_proposer.py   (LoRAProposer, arm="greedy")
    Arm 3  risk   LoRA  -> layer1/lora_proposer.py   (LoRAProposer, arm="risk")

The shared seam they all plug into:

  * Prompt    — built from the SAME `Task` (data points + grammar). Evolution adds
                best-so-far inspirations; the LoRA arms show the data only.
  * Parse     — `sia.expression.parse_expression(text) -> Node | None` turns the
                model's free-form text into the SAME `Node` the SAME `Verifier`
                scores. Invalid parse -> reward 0 (handled by the verifier).
  * ask/tell  — `ask()` samples a batch; `tell()` updates the archive (evolution)
                or takes one reward-weighted LoRA gradient step (greedy/risk).

The greedy vs. risk difference is ONLY the per-sample weight `w_i`, taken from
`sia/objectives.py` — the SAME formulas as Layer 0's `rl_greedy` / `rl_risk`.

THE TRANSFER QUESTION (what Layer 1 exists to answer):
    Does the Layer 0 ranking  (evolution ~= risk-seeking > greedy)  still hold
    with an LLM proposer? Or does the pretrained LLM prior change the dynamics --
    e.g. prevent the greedy collapse because the prior already covers diverse,
    structured expressions? See layer1/README.md.
"""
from __future__ import annotations

# Re-export the real proposers so any old import site keeps working.
from layer1.llm_evolution import LLMEvolutionProposer  # noqa: F401
from layer1.lora_proposer import INVALID, LoRAProposer  # noqa: F401
