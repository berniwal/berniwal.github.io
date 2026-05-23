"""LAYER 1 STUB -- design only, NOT implemented in v1.

This shows how an LLM proposer plugs into the EXACT SAME seams as Layer 0:
the same `Task`, the same `Verifier`, and the same `Proposer.ask()/tell()`
interface. Only the proposer changes. Nothing here imports MLX yet; every
LLM-specific step is a clearly marked TODO.

Run target hardware: M4 MacBook (64GB). Proposer model: Qwen2.5-0.5B/1.5B via
`mlx-lm`. The three Layer 1 arms mirror the three Layer 0 arms:

    Layer 0                         Layer 1 (LLM)
    ---------------------------     -------------------------------------------
    Evolution (GP)              ->  Program-database evolution (AlphaEvolve /
                                    ThetaEvolve): feed best expressions back into
                                    the prompt as inspiration. No weight updates.
    Greedy RL (E[R])            ->  Greedy LoRA fine-tune on high-reward outputs
                                    (SFT / expected-reward policy gradient).
    Risk-seeking / entropic RL  ->  Risk-seeking / entropic LoRA fine-tune
                                    (train only on the top-epsilon outputs, or the
                                    e^{beta R}-weighted outputs).

THE TRANSFER QUESTION (what Layer 1 exists to answer):
    Does the Layer 0 ranking  (evolution ~= risk-seeking > greedy)  still hold
    with an LLM proposer? Or does the pretrained LLM prior change the dynamics --
    e.g. prevent the greedy collapse because the prior already covers diverse,
    structured expressions?
"""
from __future__ import annotations

from sia.expression import Node
from sia.proposers.base import Proposer
from sia.verifier import Result


def format_prompt(task, inspirations: list[tuple[str, float]] | None = None) -> str:
    """Build the LLM prompt from the SAME task object Layer 0 uses.

    TODO: include the (x, y) data points and the grammar; for the evolution arm,
    append the best-so-far expressions in `inspirations` (expr string, reward).
    """
    raise NotImplementedError("Layer 1 stub")


def parse_expression(text: str) -> Node | None:
    """Parse the model's text output into the SAME Node representation Layer 0
    uses, so it can be scored by the SAME Verifier.

    TODO: parse infix (e.g. via `ast` with a whitelisted grammar) into Node, or
    prompt the model to emit prefix tokens and reuse `expression.from_prefix`.
    Return None on parse failure -> Verifier already maps invalid -> reward 0.
    """
    raise NotImplementedError("Layer 1 stub")


class LLMProposer(Proposer):
    """Skeleton shared by all three Layer 1 arms. Same ask/tell contract as
    Layer 0, so `runner.run_method` drives it unchanged."""

    def __init__(self, task, rng, arm: str = "evolution", model: str = "Qwen2.5-0.5B",
                 batch_size: int = 16, **hp):
        super().__init__(task, rng, **hp)
        self.arm = arm                # "evolution" | "greedy" | "risk"
        self.batch_size = batch_size
        # TODO: load the model + tokenizer via mlx_lm.load(model)
        # TODO: for greedy/risk arms, initialize a LoRA adapter + optimizer
        self.archive: list[tuple[Node, float]] = []  # program database (evolution arm)

    def ask(self) -> list[Node]:
        # TODO: build prompt(s) (with self.archive inspirations for evolution),
        #       sample `batch_size` completions via mlx-lm, parse_expression each.
        raise NotImplementedError("Layer 1 stub")

    def tell(self, candidates: list[Node], results: list[Result]) -> None:
        # evolution: update self.archive with the best (program database).
        # greedy:    LoRA step on high-reward outputs (expected-reward objective).
        # risk:      LoRA step on the top-epsilon / e^{beta R}-weighted outputs
        #            -- reuse the SAME weighting logic as Layer 0's rl_risk.
        raise NotImplementedError("Layer 1 stub")
