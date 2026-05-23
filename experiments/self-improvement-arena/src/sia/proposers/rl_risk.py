"""Risk-seeking RL: the SAME policy as greedy, but optimize the BEST outcomes
instead of the average. Only the per-trajectory weight vector changes.

Two modes (selectable in config):

- ``quantile`` (default; Deep Symbolic Regression, Petersen et al. 2021): maximize
  the (1-epsilon) reward quantile. Per batch, keep only the top-epsilon samples and
  weight them by  (R_i - R_quantile);  everything below the quantile gets weight 0.
  The quantile IS the baseline.

- ``entropic`` (Jiang et al. 2025 / TTT-Discover): maximize J_beta = (1/beta) log
  E[e^{beta R}], whose gradient is an exponentially-tilted REINFORCE with weights
  proportional to e^{beta R}. beta -> 0 recovers greedy; beta -> inf recovers max.
  beta is made scale-invariant by dividing by the batch reward std (an "adaptive
  beta" in the spirit of TTT-Discover).

See the README lineage note: the hard quantile is essentially a limiting case of
the soft tilt; DSR (2021) and the entropic line (2025) are the same idea.
"""
from __future__ import annotations

import numpy as np

from ..policy import RNNPolicy
from ..verifier import Result
from .base import Proposer


class RLRisk(Proposer):
    def __init__(self, task, rng, batch_size: int = 200, hidden: int = 32,
                 max_length: int = 24, lr: float = 0.01, ent_coef: float = 0.01,
                 mode: str = "quantile", epsilon: float = 0.1, beta: float = 2.0,
                 seed: int | None = None, **hp):
        super().__init__(task, rng, **hp)
        if mode not in ("quantile", "entropic"):
            raise ValueError(f"mode must be 'quantile' or 'entropic', got {mode!r}")
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.mode = mode
        self.epsilon = epsilon
        self.beta = beta
        seed = int(rng.integers(1 << 30)) if seed is None else seed
        self.policy = RNNPolicy(hidden=hidden, max_length=max_length, lr=lr, seed=seed)

    def ask(self):
        return self.policy.sample(self.batch_size, self.rng)

    def _weights(self, R: np.ndarray) -> np.ndarray:
        if self.mode == "quantile":
            q = np.quantile(R, 1.0 - self.epsilon)
            return np.where(R >= q, R - q, 0.0)  # train only on the top tail
        # entropic: scale-invariant exponential tilt, centered to sum ~0
        b = self.beta / (R.std() + 1e-8)
        logits = b * (R - R.max())             # subtract max for stability
        sm = np.exp(logits)
        sm /= sm.sum()
        return sm * len(R) - 1.0               # O(1) scale, mean 0

    def tell(self, candidates, results: list[Result]) -> None:
        R = np.array([r.reward for r in results])
        self.policy.reinforce(self._weights(R), self.ent_coef)

    def diagnostics(self) -> dict:
        return {"policy_entropy": self.policy.last_entropy()}
