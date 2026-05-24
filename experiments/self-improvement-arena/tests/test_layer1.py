"""Layer 1 LoRA proposer: ask/tell/budget/weighting bookkeeping, tested WITHOUT
MLX or an LLM. A fake proposer overrides the three MLX-touching methods
(_setup_model / _sample_batch / _lora_step) with canned behavior, so the
portable loop logic is exercised on any machine.

The actual MLX LoRA training step (_lora_step's real body) is NOT exercised here
-- it needs Apple Silicon; see layer1/README.md for the on-device smoke test.

Run: PYTHONPATH=src:. python -m tests.test_layer1
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from layer1.lora_proposer import INVALID, LoRAProposer  # noqa: E402
from sia.objectives import cvar_weights, greedy_weights, quantile_weights  # noqa: E402
from sia.task import make_task  # noqa: E402
from sia.verifier import Result, Verifier  # noqa: E402


class _FakeTokenizer:
    """Returns canned prompt token ids; never touches a real tokenizer."""
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True):
        # one token per character of the message is plenty for bookkeeping tests
        return list(range(1, 8))


class _FakeLoRAProposer(LoRAProposer):
    """LoRAProposer with the MLX seams stubbed out. ``canned`` is the list of
    strings the 'model' emits (cycled to fill the batch)."""

    def __init__(self, *a, canned=None, **k):
        self._canned = canned or ["x*x + sin(x)"]
        self.recorded = []          # (prompt, completions, weights) per _lora_step
        super().__init__(*a, **k)

    def _setup_model(self):
        self.opt = "fake-optimizer"     # no MLX

    def _sample_batch(self, prompt_tokens):
        out = []
        for i in range(self.batch_size):
            text = self._canned[i % len(self._canned)]
            out.append((text, [100 + i, 200 + i]))  # arbitrary non-empty token ids
        return out

    def _lora_step(self, prompt_tokens, completions, weights):
        self.recorded.append((list(prompt_tokens), [list(c) for c in completions],
                              np.asarray(weights, dtype=float)))
        self._last_loss = 0.0


def _make(canned, arm="greedy", batch_size=None, **k):
    task = make_task("medium", seed=0)
    rng = np.random.default_rng(0)
    bs = batch_size if batch_size is not None else len(canned)
    return _FakeLoRAProposer(task, rng, model=object(), tokenizer=_FakeTokenizer(),
                             arm=arm, batch_size=bs, canned=canned, **k)


def _results(rewards):
    return [Result(reward=float(r), mse=0.0, valid=r > 0, complexity=1) for r in rewards]


def test_ask_parses_and_flags_invalid():
    p = _make(["x*x + sin(x)", "not a formula!!", "x + 1"])
    cands = p.ask()
    assert len(cands) == 3
    assert cands[1] is INVALID                 # unparseable -> INVALID sentinel
    assert cands[0] is not INVALID and cands[2] is not INVALID
    assert np.isclose(p._last_valid_frac, 2 / 3)
    assert p.diagnostics()["valid_fraction"] == p._last_valid_frac


def test_invalid_candidate_scores_zero_with_real_verifier():
    p = _make(["garbage", "x*x + sin(x)"])
    ver = Verifier(make_task("medium", seed=0))
    cands = p.ask()
    res = [ver(c) for c in cands]
    assert res[0].reward == 0.0 and not res[0].valid   # INVALID -> reward 0
    assert res[1].valid


def test_ask_stashes_completions_for_tell():
    p = _make(["x", "x*x", "sin(x)"])
    p.ask()
    assert p._pending_prompt == list(range(1, 8))
    assert p._pending_completions == [[100, 200], [101, 201], [102, 202]]


def test_tell_greedy_uses_greedy_weights():
    p = _make(["x", "x*x", "sin(x)", "x + 1"], arm="greedy")
    p.ask()
    R = [0.1, 0.4, 0.2, 0.9]
    p.tell([None] * 4, _results(R))             # candidates list is unused by tell
    assert len(p.recorded) == 1
    _, completions, w = p.recorded[0]
    assert np.allclose(w, greedy_weights(np.array(R)))
    assert len(completions) == 4               # one completion per sample
    assert p._pending_completions is None      # cleared after the step


def test_tell_risk_quantile_uses_quantile_weights():
    p = _make(["x", "x*x", "sin(x)", "x + 1"], arm="risk", mode="quantile", epsilon=0.5)
    p.ask()
    R = [0.1, 0.4, 0.2, 0.9]
    p.tell([None] * 4, _results(R))
    _, _, w = p.recorded[0]
    assert np.allclose(w, quantile_weights(np.array(R), 0.5))


def test_tell_risk_cvar_uses_cvar_weights():
    p = _make(["x", "x*x", "sin(x)", "x + 1"], arm="risk", mode="cvar", epsilon=0.5)
    p.ask()
    R = [0.1, 0.4, 0.2, 0.9]
    p.tell([None] * 4, _results(R))
    _, _, w = p.recorded[0]
    assert np.allclose(w, cvar_weights(np.array(R), 0.5))
    assert (w <= 1e-12).all()                  # risk-averse weights are non-positive


def test_tell_risk_entropic_records_beta():
    p = _make(["x", "x*x", "sin(x)", "x + 1"], arm="risk", mode="entropic",
              beta_rule="fixed", beta=2.0)
    p.ask()
    p.tell([None] * 4, _results([0.1, 0.4, 0.2, 0.9]))
    assert np.isfinite(p._last_beta) and p._last_beta > 0
    assert np.isfinite(p.diagnostics()["beta"])


def test_gradient_step_does_not_cost_verifier_calls():
    """One round = batch_size verifier calls (the generations); the LoRA step adds
    none. This is the fair-budget contract the runner relies on."""
    p = _make(["x*x + sin(x)"] * 4, batch_size=4, arm="greedy")
    ver = Verifier(make_task("medium", seed=0))
    for _ in range(3):                          # three ask/tell rounds
        cands = p.ask()
        res = [ver(c) for c in cands]
        p.tell(cands, res)
    assert ver.calls == 3 * 4                   # exactly batch_size per round
    assert len(p.recorded) == 3                 # three gradient steps taken


def test_tell_without_ask_is_noop():
    p = _make(["x"], arm="greedy")
    p.tell([None], _results([0.5]))             # no preceding ask()
    assert p.recorded == []


def test_arm_validation():
    task = make_task("medium", seed=0)
    rng = np.random.default_rng(0)
    for bad in [dict(arm="nope"), dict(arm="risk", mode="bad"),
                dict(arm="risk", beta_rule="bad")]:
        try:
            _FakeLoRAProposer(task, rng, model=object(), tokenizer=_FakeTokenizer(),
                              batch_size=2, canned=["x"], **bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


def test_constfit_recovers_structure():
    """The BFGS reward: a C-skeleton the LLM would propose, fit to the data, recovers
    the target (no MLX needed -- just sia + scipy)."""
    from layer1.constfit import parse_and_fit
    from sia.task import make_task
    from sia.verifier import Verifier
    task = make_task("medium", seed=0)          # x^2 + sin(x)
    # the LLM's structure with C placeholders; BFGS should fit C -> (1, 1, 0)
    node = parse_and_fit("C*x*x + C*sin(x) + C", task.x_train, task.y_train)
    assert node is not None
    assert Verifier(task).success(node)          # held-out recovery
    # a skeleton with a spurious extra term still recovers (cos coefficient -> ~0)
    node2 = parse_and_fit("C*x*x + C*sin(x) + C*cos(x)", task.x_train, task.y_train)
    assert node2 is not None and Verifier(task).success(node2)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
