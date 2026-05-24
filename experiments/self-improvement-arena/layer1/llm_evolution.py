"""Layer 1, arm 1: LLM program-database evolution (AlphaEvolve / ThetaEvolve style).

The proposer is a small instruction-tuned LLM (via mlx-lm). It keeps an *archive*
(program database) of the best expressions found so far and, each round, prompts the
model with the data points plus those best expressions as inspiration, asking for a
better one. No weight updates — improvement comes from the in-context feedback loop.

It plugs into the SAME seam as Layer 0: `ask()` proposes a batch of `Node`s,
`tell()` learns (here: updates the archive), and the SAME `Verifier` scores them.
mlx-lm is imported lazily so importing this module never burdens Layer 0.
"""
from __future__ import annotations

import numpy as np

# Layer 0 core (the shared task/verifier/representation), imported via the src path.
from sia.expression import Node, leaf, parse_expression, to_infix
from sia.proposers.base import Proposer
from sia.verifier import Result

# A candidate that always scores 0 (1/(1+MSE) on a NaN -> invalid). Used when the
# model's output can't be parsed: the wasted generation still costs one verifier
# call (fair budgeting), but earns nothing.
INVALID = Node("/", [leaf("0.0"), leaf("0.0")])


def load_model(model_id: str):
    """Load an mlx-lm model + tokenizer once (lazy import)."""
    from mlx_lm import load
    return load(model_id)


class LLMEvolutionProposer(Proposer):
    def __init__(self, task, rng, model, tokenizer, batch_size: int = 8,
                 archive_size: int = 12, n_inspirations: int = 5,
                 temperature: float = 0.8, max_tokens: int = 48,
                 n_data_shown: int = 12, use_archive: bool = True,
                 const_placeholder: bool = False, **hp):
        super().__init__(task, rng, **hp)
        self.model = model
        self.tok = tokenizer
        self.batch_size = batch_size
        # DSR-style constant placeholder: prompt the model to use `C` for numbers and
        # score with all constants = 1 (it proposes structure, not coefficients).
        self.const_placeholder = const_placeholder
        self.archive_size = archive_size
        self.n_inspirations = n_inspirations
        # use_archive=False -> the prompt is data-only and nothing is fed back:
        # this turns the arm into the BEST-OF-N control (frozen weights, no
        # in-context learning) -- the shared floor for evolution and the LoRA arms.
        self.use_archive = use_archive
        self.max_tokens = max_tokens
        self.n_data_shown = n_data_shown
        from mlx_lm.sample_utils import make_sampler
        self._sampler = make_sampler(temp=temperature)
        # (reward, mse, infix, Node), kept sorted best-first (highest reward)
        self.archive: list[tuple[float, float, str, Node]] = []
        self._last_valid_frac = float("nan")
        self._last_prompt = ""               # decoded prompt text (for the app)
        self._last_responses: list[str] = []  # raw model outputs (for the app)

    # --- prompt construction -------------------------------------------------
    def _data_block(self) -> str:
        x, y = self.task.x_train, self.task.y_train
        idx = np.linspace(0, len(x) - 1, min(self.n_data_shown, len(x))).astype(int)
        return "\n".join(f"  x = {x[i]:+.3f}   y = {y[i]:+.3f}" for i in idx)

    def _prompt(self) -> str:
        if self.const_placeholder:
            vocab = ("Allowed: the variable x, operators + - * /, the functions sin and "
                     "cos, and the constant placeholder C. Write C in place of every "
                     "numeric constant or coefficient -- for example write C*x + C "
                     "instead of 2.5*x + 1.3. Each C's value is chosen automatically.")
        else:
            vocab = ("Allowed: the variable x, operators + - * /, the functions sin and "
                     "cos, and numeric constants.")
        rules = ("You are doing symbolic regression. Find a formula y = f(x) that fits "
                 f"the data.\n{vocab}\nThe data may be nonlinear or "
                 "periodic — consider terms like x*x, x*x*x, or sin/cos, not just "
                 "straight lines.\nReply with ONLY the formula for f(x) on a single "
                 "line — no words, no 'y =', no code fences.")
        msg = f"{rules}\n\nData points:\n{self._data_block()}\n"
        if self.use_archive and self.archive:  # program-database inspiration (the evolution signal)
            best = self.archive[: self.n_inspirations]
            lines = "\n".join(f"  {infix}   (squared error {mse:.3f}, lower is better)"
                              for _, mse, infix, _ in best)
            msg += ("\nYour best formulas so far (with their error). Propose a formula "
                    "that lowers the error — keep the parts that help and try a "
                    f"structurally different idea if the error is large:\n{lines}\n")
        msg += "\nNew formula for f(x):"
        self._last_prompt = msg  # surfaced to the app for inspection
        return self.tok.apply_chat_template(
            [{"role": "user", "content": msg}], add_generation_prompt=True)

    # --- the ask / tell seam -------------------------------------------------
    def ask(self) -> list[Node]:
        from mlx_lm import generate
        prompt = self._prompt()
        cands, raw, n_valid = [], [], 0
        for _ in range(self.batch_size):
            text = generate(self.model, self.tok, prompt=prompt,
                            max_tokens=self.max_tokens, sampler=self._sampler,
                            verbose=False)
            raw.append(text)
            node = parse_expression(text, const_placeholder=self.const_placeholder)
            if node is None:
                cands.append(INVALID)
            else:
                cands.append(node)
                n_valid += 1
        self._last_valid_frac = n_valid / max(self.batch_size, 1)
        self._last_responses = raw  # raw outputs for the app
        return cands

    def tell(self, candidates: list[Node], results: list[Result]) -> None:
        for c, r in zip(candidates, results):
            if c is INVALID or not r.valid:
                continue
            self.archive.append((r.reward, r.mse, to_infix(c), c))
        # dedup by expression string, keep the best `archive_size` (highest reward)
        seen, kept = set(), []
        for item in sorted(self.archive, key=lambda t: t[0], reverse=True):
            if item[2] in seen:
                continue
            seen.add(item[2])
            kept.append(item)
            if len(kept) >= self.archive_size:
                break
        self.archive = kept

    def diagnostics(self) -> dict:
        return {"policy_entropy": float("nan"),
                "valid_fraction": self._last_valid_frac,
                "archive_best": self.archive[0][0] if self.archive else 0.0}

    def last_io(self) -> dict:
        """The prompt sent and the raw responses from the last ask() -- for the app
        to display what the LLM actually saw and produced."""
        return {"prompt": self._last_prompt, "responses": list(self._last_responses)}
