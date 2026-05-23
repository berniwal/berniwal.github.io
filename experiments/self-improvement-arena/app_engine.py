"""Engine behind the Streamlit visualizer -- the stateful, steppable core, with
NO Streamlit and NO MLX import at module load. Keeping the logic here (rather than
inside the Streamlit script) means it is unit-testable on any machine
(``tests/test_app_engine.py``), and the UI in ``streamlit_app.py`` stays a thin
view over it.

A "step" is one batch of the SAME ask/tell loop the runner uses:

    cands = proposer.ask(); results = [verifier(c) ...]; proposer.tell(...)

We hold the live ``Proposer`` + ``Verifier`` per method and append per-step
metrics (best-so-far reward, batch diversity, policy entropy / valid fraction) so
the app can redraw the dynamics after each click. Layer 0 methods run instantly;
the Layer 1 (LLM) builder is here too but is only ever exercised on Apple Silicon.
"""
from __future__ import annotations

import importlib.util
import platform
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Self-bootstrap sys.path so `import sia` / `import layer1` work whether or not the
# package is pip-installed, and regardless of cwd (mirrors run_layer1.py).
ROOT = Path(__file__).resolve().parent
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sia.expression import Node, evaluate, to_infix, to_prefix  # noqa: E402
from sia.metrics import unique_fraction  # noqa: E402
from sia.proposers import get_proposer  # noqa: E402
from sia.task import make_task  # noqa: E402
from sia.verifier import Verifier  # noqa: E402

# Layer 0 method presets, keyed to match sia.plotting.STYLE (so colors/labels are
# shared with the offline figures). `hp` excludes batch_size, which the app sets.
LAYER0_PRESETS: dict[str, dict] = {
    "random": dict(proposer="random", hp=dict(max_depth=4)),
    "gp": dict(proposer="gp", hp=dict(tournament_size=5, crossover_rate=0.6,
                                      mutation_rate=0.3, immigrant_rate=0.1,
                                      max_depth=4, max_complexity=30)),
    "greedy": dict(proposer="greedy", hp=dict(hidden=32, max_length=24, lr=0.01,
                                              ent_coef=0.01)),
    "risk": dict(proposer="risk", hp=dict(hidden=32, max_length=24, lr=0.01,
                                          ent_coef=0.01, mode="quantile", epsilon=0.1)),
    "risk_entropic": dict(proposer="risk", hp=dict(hidden=32, max_length=24, lr=0.01,
                                                   ent_coef=0.01, mode="entropic",
                                                   beta_rule="fixed", beta=2.0)),
}

LAYER1_ARMS = ("evolution", "greedy_lora", "risk_lora")


@dataclass
class MethodState:
    """All the live state + per-step history for ONE method's run."""
    key: str
    proposer: object
    verifier: Verifier
    steps: int = 0
    calls: list[int] = field(default_factory=list)        # cumulative verifier calls
    best_reward: list[float] = field(default_factory=list)  # best-so-far per step
    diversity: list[float] = field(default_factory=list)    # unique fraction per batch
    entropy: list[float] = field(default_factory=list)      # policy entropy (NaN if n/a)
    valid_frac: list[float] = field(default_factory=list)   # LLM valid parse frac (NaN if n/a)
    best: float = 0.0
    best_expr: Node | None = None
    success: bool = False
    evals_to_solve: int | None = None
    last_batch: list[Node] = field(default_factory=list)
    last_rewards: list[float] = field(default_factory=list)


def build_layer0_state(key: str, task, seed: int, batch_size: int) -> MethodState:
    """Instantiate one Layer 0 method (fresh proposer + its own verifier) on the
    shared task, seeded for reproducibility."""
    preset = LAYER0_PRESETS[key]
    rng = np.random.default_rng(seed)
    hp = dict(preset["hp"], batch_size=batch_size)
    if preset["proposer"] == "gp":
        hp["pop_size"] = batch_size  # keep population == batch for a clean budget
    proposer = get_proposer(preset["proposer"])(task, rng, **hp)
    return MethodState(key=key, proposer=proposer, verifier=Verifier(task))


def build_layer1_state(arm: str, task, seed: int, model, tokenizer,
                       batch_size: int = 8, max_tokens: int = 48, **hp) -> MethodState:
    """Instantiate one Layer 1 (LLM) arm. Imports the proposers lazily -- and the
    proposers themselves import MLX lazily -- so this is only reachable on a box
    where mlx-lm is installed (Apple Silicon). NOT exercised in CI on Linux."""
    if arm not in LAYER1_ARMS:
        raise ValueError(f"unknown arm {arm!r}; choices: {list(LAYER1_ARMS)}")
    from layer1.llm_evolution import LLMEvolutionProposer
    from layer1.lora_proposer import LoRAProposer

    rng = np.random.default_rng(seed)
    common = dict(batch_size=batch_size, max_tokens=max_tokens)
    if arm == "evolution":
        proposer = LLMEvolutionProposer(task, rng, model, tokenizer, **common, **hp)
    elif arm == "greedy_lora":
        proposer = LoRAProposer(task, rng, model, tokenizer, arm="greedy", **common, **hp)
    else:  # risk_lora
        proposer = LoRAProposer(task, rng, model, tokenizer, arm="risk", **common, **hp)
    return MethodState(key=arm, proposer=proposer, verifier=Verifier(task))


def step(state: MethodState, n_steps: int = 1) -> None:
    """Advance one method by ``n_steps`` batches, recording metrics each step.
    This is exactly the runner's inner loop; success/evals-to-solve are exact."""
    for _ in range(n_steps):
        cands = state.proposer.ask()
        results = [state.verifier(c) for c in cands]
        state.proposer.tell(cands, results)
        state.steps += 1
        for c, r in zip(cands, results):
            if r.reward > state.best:
                state.best, state.best_expr = r.reward, c
            if state.evals_to_solve is None and state.verifier.success(c):
                state.success = True
                state.evals_to_solve = state.verifier.calls
        state.calls.append(state.verifier.calls)
        state.best_reward.append(state.best)
        state.diversity.append(unique_fraction(cands))
        diag = (state.proposer.diagnostics()
                if hasattr(state.proposer, "diagnostics") else {})
        state.entropy.append(float(diag.get("policy_entropy", float("nan"))))
        state.valid_frac.append(float(diag.get("valid_fraction", float("nan"))))
        state.last_batch = cands
        state.last_rewards = [r.reward for r in results]


def fit_overlay(task, expr: Node | None, n: int = 200):
    """Smooth curves for the 'is it fitting the data' plot: the true target f(x)
    and (if any) the current best expression, both over the training x-range.
    Non-finite predictions become NaN so matplotlib just leaves a gap."""
    lo, hi = float(task.x_train.min()), float(task.x_train.max())
    xs = np.linspace(lo, hi, n)
    with np.errstate(all="ignore"):
        y_target = evaluate(task.target_expr, xs)
    y_pred = None
    if expr is not None:
        with np.errstate(all="ignore"):
            y = evaluate(expr, xs)
        y_pred = np.where(np.isfinite(y), y, np.nan)
    return xs, y_target, y_pred


def batch_summary(state: MethodState, k: int = 8) -> list[tuple[str, int, float]]:
    """Most frequent expressions in the latest batch as ``(infix, count, reward)``.
    When a policy mode-collapses, one row's count approaches the whole batch -- the
    collapse made legible."""
    counts = Counter(tuple(to_prefix(c)) for c in state.last_batch)
    rep: dict[tuple, tuple[str, float]] = {}
    for c, r in zip(state.last_batch, state.last_rewards):
        key = tuple(to_prefix(c))
        if key not in rep:
            rep[key] = (to_infix(c), r)
    return [(rep[key][0], n, rep[key][1]) for key, n in counts.most_common(k)]


# --- Layer 1 capability detection (runtime, not install-time) ----------------
def mlx_available() -> bool:
    """True if mlx-lm can be imported here. The Streamlit app uses this to enable
    or disable the Layer 1 controls; off-Mac it stays False and the app says so."""
    return importlib.util.find_spec("mlx_lm") is not None


def platform_label() -> str:
    return f"{platform.system().lower()}/{platform.machine()}"
