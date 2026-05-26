"""One-shot CoT inspector: generate ~5 reasoning rollouts with Qwen3-1.7B on the
'harder' target, print each raw text + parse result. No training, no LoRA grad.

Used to diagnose why the smoke pod's valid_fraction was 0-38%: are the rollouts
actually emitting a parseable formula, or is the prompt / extractor wrong?
"""
from __future__ import annotations

import json
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sia.expression import to_infix  # noqa: E402
from sia.task import make_task  # noqa: E402
from layer1.torch_lora_proposer import TorchLoRAProposer  # noqa: E402


def main():
    task = make_task("harder", n_points=20, x_range=(-1.0, 1.0), seed=0)
    # Budgeted reasoning: generous 2048 thinking tokens (sample 3 of the 768-budget
    # run hit a good shape near the upper end of its trace, so we give it more
    # room), then forced wrap-up sentence + 96 tokens for the formula.
    prop = TorchLoRAProposer(
        task, model_id="Qwen/Qwen3-1.7B", arm="risk", mode="quantile",
        batch_size=5, micro_batch=1, max_new_tokens=2048, temperature=1.0,
        reasoning=True, thinking_budget=2048, answer_budget=96, seed=0,
    )

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prop._prompt_text())
    print()

    cands = prop.ask()
    io = prop.last_io()

    out = {"prompt": io["prompt"], "samples": []}
    for i, (text, c) in enumerate(zip(io["responses"], cands)):
        extracted = prop._extract_formula(text)
        parsed = None if c.op == "1.0" else to_infix(c)
        print("=" * 80)
        print(f"SAMPLE {i}  parsed_ok={parsed is not None}")
        print("=" * 80)
        print("RAW (first 2000 chars):")
        print(text[:2000])
        if len(text) > 2000:
            print(f"... [{len(text) - 2000} more chars]")
        print()
        print(f"EXTRACTED FINAL LINE: {extracted!r}")
        print(f"PARSED EXPRESSION:    {parsed!r}")
        print()
        out["samples"].append(dict(
            raw=text, extracted=extracted, parsed=parsed,
            raw_len=len(text), has_think_close="</think>" in text,
        ))

    os.makedirs("results/cot", exist_ok=True)
    with open("results/cot/inspection.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote results/cot/inspection.json ({len(out['samples'])} samples)")


if __name__ == "__main__":
    main()
