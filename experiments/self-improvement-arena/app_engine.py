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

import concurrent.futures
import importlib.util
import platform
import sys
import threading
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
    "cvar": dict(proposer="risk", hp=dict(hidden=32, max_length=24, lr=0.01,
                                          ent_coef=0.01, mode="cvar", epsilon=0.1)),
    "greedy": dict(proposer="greedy", hp=dict(hidden=32, max_length=24, lr=0.01,
                                              ent_coef=0.01)),
    "risk": dict(proposer="risk", hp=dict(hidden=32, max_length=24, lr=0.01,
                                          ent_coef=0.01, mode="quantile", epsilon=0.1)),
    "risk_entropic": dict(proposer="risk", hp=dict(hidden=32, max_length=24, lr=0.01,
                                                   ent_coef=0.01, mode="entropic",
                                                   beta_rule="fixed", beta=2.0)),
}

LAYER1_ARMS = ("best_of_n", "evolution", "greedy_lora", "risk_lora")


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
    if arm == "best_of_n":  # control: evolution minus the archive (no learning)
        proposer = LLMEvolutionProposer(task, rng, model, tokenizer,
                                        use_archive=False, **common, **hp)
    elif arm == "evolution":
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


def last_io(state: MethodState) -> dict:
    """Prompt + raw responses from the proposer's last ask() (Layer 1 only).
    Empty for Layer 0 proposers, which have no text I/O."""
    p = state.proposer
    return p.last_io() if hasattr(p, "last_io") else {"prompt": "", "responses": []}


def _batch_rows(cands, rewards, k: int = 8) -> list[tuple[str, int, float]]:
    """Most frequent expressions in a batch as (infix, count, reward)."""
    counts = Counter(tuple(to_prefix(c)) for c in cands)
    rep: dict[tuple, tuple[str, float]] = {}
    for c, r in zip(cands, rewards):
        key = tuple(to_prefix(c))
        if key not in rep:
            rep[key] = (to_infix(c), r)
    return [(rep[key][0], n, rep[key][1]) for key, n in counts.most_common(k)]


@dataclass
class Frame:
    """A snapshot of one batch in a full run, enough to replay/inspect it later."""
    batch: int
    calls: int
    best: float                       # best-so-far reward up to this batch
    best_infix: str
    best_expr: Node | None             # for the fit overlay
    batch_mean: float                  # this batch's mean reward (the learning signal)
    valid_frac: float
    rows: list                         # (infix, count, reward) for this batch
    prompt: str
    responses: list


@dataclass
class RunReplay:
    """A whole Layer-1 run, recorded batch-by-batch so the UI can step through it
    after all the (slow) computation is done."""
    arm: str
    target: str
    budget: int
    batch_size: int
    frames: list                       # list[Frame]


def run_batch(state: MethodState) -> Frame:
    """Run ONE batch on an existing Layer-1 MethodState and return its Frame
    (best-so-far, this batch's proposals, prompt + raw responses). Lets a UI drive
    the loop -- one mlx_call per batch -- so it can show a progress bar between
    batches. ``run_full`` just calls this in a loop."""
    ver, proposer = state.verifier, state.proposer
    cands = proposer.ask()
    results = [ver(c) for c in cands]
    proposer.tell(cands, results)
    rewards = [r.reward for r in results]
    for c, r in zip(cands, rewards):
        if r > state.best:
            state.best, state.best_expr = r, c
    state.steps += 1
    diag = proposer.diagnostics() if hasattr(proposer, "diagnostics") else {}
    io = proposer.last_io() if hasattr(proposer, "last_io") else {"prompt": "", "responses": []}
    return Frame(
        batch=state.steps, calls=ver.calls, best=state.best,
        best_infix=to_infix(state.best_expr) if state.best_expr is not None else "",
        best_expr=state.best_expr,
        batch_mean=float(np.mean(rewards)) if rewards else 0.0,
        valid_frac=float(diag.get("valid_fraction", float("nan"))),
        rows=_batch_rows(cands, rewards), prompt=io["prompt"], responses=io["responses"])


def run_full(arm: str, task, seed: int, model, tokenizer, budget: int,
             batch_size: int = 8, max_tokens: int = 48, **hp) -> RunReplay:
    """Run one Layer-1 arm to ``budget`` verifier calls, recording a Frame per batch.
    Convenience wrapper for headless use; the UI builds the state and calls
    ``run_batch`` per batch directly so it can render a progress bar."""
    state = build_layer1_state(arm, task, seed, model, tokenizer,
                               batch_size=batch_size, max_tokens=max_tokens, **hp)
    frames: list[Frame] = []
    while state.verifier.calls < budget:
        frames.append(run_batch(state))
    return RunReplay(arm=arm, target=task.name, budget=budget,
                     batch_size=batch_size, frames=frames)


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


# --- MLX threading shim (required under Streamlit) ---------------------------
# MLX streams are THREAD-LOCAL, but Streamlit dispatches each rerun on a thread
# from a pool -- so loading the model on one rerun's thread and generating on the
# next raises "There is no Stream(gpu, N) in current thread." Funnelling every
# MLX-touching call (model load, LoRA attach, generate, gradient step) through one
# persistent single worker thread keeps them all on one stream. Layer 0 never uses
# this. The pool is created lazily so importing this module stays MLX-free.
_mlx_pool = None
_mlx_lock = threading.Lock()


def mlx_call(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` on the dedicated MLX worker thread and return
    its result. Use for ALL MLX-touching work in the Streamlit app so model load
    and stepping share one thread (and thus one MLX stream)."""
    global _mlx_pool
    with _mlx_lock:
        if _mlx_pool is None:
            _mlx_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mlx")
    return _mlx_pool.submit(fn, *args, **kwargs).result()


# --- Layer 1 capability detection (runtime, not install-time) ----------------
def mlx_available() -> bool:
    """True if mlx-lm can be imported here. The Streamlit app uses this to enable
    or disable the Layer 1 controls; off-Mac it stays False and the app says so."""
    return importlib.util.find_spec("mlx_lm") is not None


def platform_label() -> str:
    return f"{platform.system().lower()}/{platform.machine()}"
