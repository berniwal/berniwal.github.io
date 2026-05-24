"""The verifier: the single, fixed reward signal shared by every method and both
layers. It is microsecond-fast and counts its own calls so the runner can give
every method exactly the same evaluation budget.

    reward_mode="mse"   : reward = 1/(1 + MSE_train)            - length_penalty * complexity
    reward_mode="nrmse" : reward = 1/(1 + RMSE_train / std(y))  - length_penalty * complexity
    invalid expression (non-finite output) -> reward 0

NRMSE (root-MSE normalized by the target's std, as in Deep Symbolic Regression) is
scale-invariant: "no better than predicting the mean" -> reward 0.5 on every target,
and it gives a gentler, more informative gradient than raw MSE. Success is judged on
a *held-out* set (held_out MSE < eps_success), independent of reward_mode, so success
rates stay comparable across modes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expression import Node, complexity, evaluate
from .task import Task


@dataclass
class Result:
    reward: float
    mse: float
    valid: bool
    complexity: int


class Verifier:
    def __init__(self, task: Task, length_penalty: float = 0.001,
                 eps_success: float = 1e-6, reward_mode: str = "mse"):
        if reward_mode not in ("mse", "nrmse"):
            raise ValueError(f"reward_mode must be 'mse' or 'nrmse', got {reward_mode!r}")
        self.task = task
        self.length_penalty = length_penalty
        self.eps_success = eps_success
        self.reward_mode = reward_mode
        self.y_std = float(np.std(task.y_train))  # for NRMSE normalization
        self.calls = 0

    def __call__(self, node: Node) -> Result:
        self.calls += 1
        comp = complexity(node)
        with np.errstate(all="ignore"):
            y = evaluate(node, self.task.x_train)
            if not np.all(np.isfinite(y)):
                return Result(0.0, float("inf"), False, comp)
            mse = float(np.mean((y - self.task.y_train) ** 2))
        if not np.isfinite(mse):
            return Result(0.0, float("inf"), False, comp)
        if self.reward_mode == "nrmse":
            fit = 1.0 / (1.0 + np.sqrt(mse) / (self.y_std + 1e-12))
        else:
            fit = 1.0 / (1.0 + mse)
        reward = fit - self.length_penalty * comp
        return Result(max(reward, 0.0), mse, True, comp)

    def success(self, node: Node) -> bool:
        """True if the candidate recovers the target on held-out data."""
        with np.errstate(all="ignore"):
            y = evaluate(node, self.task.x_heldout)
            if not np.all(np.isfinite(y)):
                return False
            mse = float(np.mean((y - self.task.y_heldout) ** 2))
        return np.isfinite(mse) and mse < self.eps_success
