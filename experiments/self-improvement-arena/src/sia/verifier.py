"""The verifier: the single, fixed reward signal shared by every method and both
layers. It is microsecond-fast and counts its own calls so the runner can give
every method exactly the same evaluation budget.

    reward = 1 / (1 + MSE_train)  -  length_penalty * complexity
    invalid expression (non-finite output) -> reward 0

Success is judged on a *held-out* set (held_out MSE < eps_success) so it measures
recovering the function, not fitting the training x's.
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
                 eps_success: float = 1e-6):
        self.task = task
        self.length_penalty = length_penalty
        self.eps_success = eps_success
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
        reward = 1.0 / (1.0 + mse) - self.length_penalty * comp
        return Result(max(reward, 0.0), mse, True, comp)

    def success(self, node: Node) -> bool:
        """True if the candidate recovers the target on held-out data."""
        with np.errstate(all="ignore"):
            y = evaluate(node, self.task.x_heldout)
            if not np.all(np.isfinite(y)):
                return False
            mse = float(np.mean((y - self.task.y_heldout) ** 2))
        return np.isfinite(mse) and mse < self.eps_success
