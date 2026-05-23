"""Greedy RL: REINFORCE maximizing EXPECTED reward.

weight_i = R_i - mean(R)   (mean baseline for variance reduction)
plus an entropy bonus so it explores -- making the comparison fair, not a
strawman. This is the arm expected to mode-collapse: optimizing the *average*
sampled expression rewards a simple-but-wrong attractor over the rare exact hit.
"""
from __future__ import annotations

import numpy as np

from ..policy import RNNPolicy
from ..verifier import Result
from .base import Proposer


class RLGreedy(Proposer):
    def __init__(self, task, rng, batch_size: int = 200, hidden: int = 32,
                 max_length: int = 24, lr: float = 0.01, ent_coef: float = 0.01,
                 seed: int | None = None, **hp):
        super().__init__(task, rng, **hp)
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        seed = int(rng.integers(1 << 30)) if seed is None else seed
        self.policy = RNNPolicy(hidden=hidden, max_length=max_length, lr=lr, seed=seed)

    def ask(self):
        return self.policy.sample(self.batch_size, self.rng)

    def tell(self, candidates, results: list[Result]) -> None:
        R = np.array([r.reward for r in results])
        weights = R - R.mean()  # expected-reward objective
        self.policy.reinforce(weights, self.ent_coef)

    def diagnostics(self) -> dict:
        return {"policy_entropy": self.policy.last_entropy()}
