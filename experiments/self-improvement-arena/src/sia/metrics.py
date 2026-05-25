"""Metrics: per-batch diversity and cross-seed aggregation onto a common grid."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .expression import Node, to_prefix


def unique_fraction(candidates: list[Node]) -> float:
    """Fraction of distinct expressions in a batch. Drops toward 1/batch when a
    policy mode-collapses; stays near 1 for random/diverse search."""
    seen = {tuple(to_prefix(c)) for c in candidates}
    return len(seen) / max(len(candidates), 1)


@dataclass
class RunLog:
    method: str
    target: str
    seed: int
    budget: int
    calls: list = field(default_factory=list)        # cumulative verifier calls
    best_reward: list = field(default_factory=list)   # best-so-far reward
    diversity: list = field(default_factory=list)      # unique fraction per batch
    policy_entropy: list = field(default_factory=list)  # NaN if not an RL policy
    success: bool = False               # NUMERIC recovery: held-out MSE < eps_success
    evals_to_solve: int | None = None
    # SYMBOLIC recovery: best expr is exactly symbolically equivalent to the target
    # (SymPy), DSR's strict definition. Only tracked when run with track_symbolic=True;
    # left False/None otherwise (so old logs still load).
    success_symbolic: bool = False
    evals_to_solve_symbolic: int | None = None
    best_expr: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def best_curve(log: RunLog, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, log.calls, log.best_reward,
                     left=log.best_reward[0], right=log.best_reward[-1])


def diversity_curve(log: RunLog, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, log.calls, log.diversity,
                     left=log.diversity[0], right=log.diversity[-1])


def aggregate(logs: list[RunLog], grid: np.ndarray, curve_fn=best_curve):
    """Mean and std of a per-run curve across seeds, on a shared call grid."""
    curves = np.stack([curve_fn(lg, grid) for lg in logs])
    return curves.mean(0), curves.std(0)


def success_rate(logs: list[RunLog]) -> float:
    return float(np.mean([lg.success for lg in logs]))


def success_rate_at(logs: list[RunLog], budget: int) -> float:
    """Fraction of seeds that had recovered the formula by `budget` verifier calls.
    Exact (uses evals_to_solve), independent of the curve logging stride."""
    return float(np.mean([
        lg.evals_to_solve is not None and lg.evals_to_solve <= budget
        for lg in logs]))


def best_reward_at(logs: list[RunLog], budget: int) -> tuple[float, float]:
    """Mean and std of best-so-far reward at `budget`, across seeds."""
    grid = np.array([budget])
    vals = np.array([best_curve(lg, grid)[0] for lg in logs])
    return float(vals.mean()), float(vals.std())


def median_evals_to_solve(logs: list[RunLog]) -> float | None:
    vals = [lg.evals_to_solve for lg in logs if lg.evals_to_solve is not None]
    return float(np.median(vals)) if vals else None
