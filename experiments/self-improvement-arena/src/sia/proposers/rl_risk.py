"""Risk-seeking RL: the SAME policy as greedy, but optimize the BEST outcomes
instead of the average. Only the per-trajectory weight vector changes.

Three modes (selectable in config):

- ``quantile`` (default; Deep Symbolic Regression, Petersen et al. 2021): maximize
  the (1-epsilon) reward quantile. Per batch, keep only the top-epsilon samples and
  weight them by  (R_i - R_quantile);  everything below the quantile gets weight 0.
  The quantile IS the baseline.

- ``cvar`` (risk-AVERSE; Tamar et al. 2014 / EPOpt, Rajeswaran et al. 2016): the
  exact mirror of ``quantile`` -- keep the WORST epsilon-tail and weight by
  (R_i - R_quantile_epsilon), which is <= 0, pushing probability away from the
  catastrophic tail. This is the original lower-tail CVaR objective that DSR
  inverts to get its risk-seeking quantile arm. Included as the deliberately
  "wrong direction" baseline for discovery (it should under-perform greedy).

- ``entropic`` (Jiang et al. 2025 / TTT-Discover): maximize J_beta = (1/beta) log
  E[e^{beta R}], whose gradient is an exponentially-tilted REINFORCE with weights
  proportional to e^{beta R}. beta -> 0 recovers greedy; beta -> inf recovers max.

For ``entropic`` the temperature beta is chosen each batch by ``beta_rule``:

- ``fixed``: beta = beta / std(R)  (constant tilt strength, scale-normalized).
  Jiang et al. (2025) use a constant beta.
- ``ess``: pick beta so the exponential weights have a target effective sample
  size (a standard importance-sampling self-tuning rule).
- ``kl``: pick beta so the induced (reward-tilted) distribution sits a target KL
  away from the batch's sampling distribution. This is our reading of
  TTT-Discover's adaptive beta -- they "set beta(s) adaptively per state by
  constraining the KL divergence of the induced policy" (Appendix A.1). NOTE: we
  constrain KL(tilted || empirical-sampling) = log B - H(weights), which is batch-
  computable and lr-independent; the paper's exact functional may differ.

See the README lineage note: the hard quantile is essentially a limiting case of
the soft tilt; DSR (2021) and the entropic line (2025) are the same idea.
"""
from __future__ import annotations

import numpy as np

from ..objectives import cvar_weights, entropic_weights, quantile_weights
from ..policy import RNNPolicy
from ..verifier import Result
from .base import Proposer


class RLRisk(Proposer):
    def __init__(self, task, rng, batch_size: int = 200, hidden: int = 32,
                 max_length: int = 24, lr: float = 0.01, ent_coef: float = 0.01,
                 mode: str = "quantile", epsilon: float = 0.1, beta: float = 2.0,
                 beta_rule: str = "fixed", target_ess: float = 0.3,
                 target_kl: float = 1.0, constraints: bool = False,
                 min_length: int = 4, entropy_gamma: float = 1.0,
                 seed: int | None = None, **hp):
        super().__init__(task, rng, **hp)
        if mode not in ("quantile", "entropic", "cvar"):
            raise ValueError(f"mode must be quantile/entropic/cvar, got {mode!r}")
        if beta_rule not in ("fixed", "ess", "kl"):
            raise ValueError(f"beta_rule must be fixed/ess/kl, got {beta_rule!r}")
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.mode = mode
        self.epsilon = epsilon
        self.beta = beta
        self.beta_rule = beta_rule    # how the entropic temperature is chosen each batch
        self.target_ess = target_ess  # for beta_rule="ess": ESS as a fraction of B
        self.target_kl = target_kl    # for beta_rule="kl": target KL in nats
        self._last_beta = float("nan")
        seed = int(rng.integers(1 << 30)) if seed is None else seed
        self.policy = RNNPolicy(hidden=hidden, max_length=max_length, lr=lr, seed=seed,
                                constraints=constraints, min_length=min_length,
                                entropy_gamma=entropy_gamma)

    def ask(self):
        return self.policy.sample(self.batch_size, self.rng)

    def _weights(self, R: np.ndarray) -> np.ndarray:
        """Per-sample weights from the shared objectives module (the SAME formulas
        Layer 1's LoRA arms use)."""
        if self.mode == "quantile":
            return quantile_weights(R, self.epsilon)
        if self.mode == "cvar":  # risk-averse lower-tail mirror of quantile
            return cvar_weights(R, self.epsilon)
        w, b = entropic_weights(R, self.beta_rule, self.beta,
                                self.target_ess, self.target_kl)
        self._last_beta = b
        return w

    def tell(self, candidates, results: list[Result]) -> None:
        R = np.array([r.reward for r in results])
        self.policy.reinforce(self._weights(R), self.ent_coef)

    def diagnostics(self) -> dict:
        return {"policy_entropy": self.policy.last_entropy(),
                "beta": self._last_beta}
