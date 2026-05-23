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

from ..policy import RNNPolicy
from ..verifier import Result
from .base import Proposer


class RLRisk(Proposer):
    def __init__(self, task, rng, batch_size: int = 200, hidden: int = 32,
                 max_length: int = 24, lr: float = 0.01, ent_coef: float = 0.01,
                 mode: str = "quantile", epsilon: float = 0.1, beta: float = 2.0,
                 beta_rule: str = "fixed", target_ess: float = 0.3,
                 target_kl: float = 1.0, seed: int | None = None, **hp):
        super().__init__(task, rng, **hp)
        if mode not in ("quantile", "entropic"):
            raise ValueError(f"mode must be 'quantile' or 'entropic', got {mode!r}")
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
        self.policy = RNNPolicy(hidden=hidden, max_length=max_length, lr=lr, seed=seed)

    def ask(self):
        return self.policy.sample(self.batch_size, self.rng)

    @staticmethod
    def _bisect_beta(Rs: np.ndarray, stat, target: float, increasing: bool) -> float:
        """Find beta >= 0 such that stat(weights(beta)) == target, where weights =
        softmax(beta * Rs) and `stat` is monotone in beta. `increasing` says whether
        stat grows with beta (KL) or shrinks (ESS)."""
        def val(b: float) -> float:
            w = np.exp(b * Rs)
            w /= w.sum()
            return stat(w)

        lo, hi = 0.0, 1.0
        # grow hi until it brackets the target
        while ((val(hi) < target) if increasing else (val(hi) > target)) and hi < 1e7:
            hi *= 2.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            below = val(mid) < target
            if below == increasing:   # need larger beta
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _entropic_beta(self, R: np.ndarray) -> float:
        if R.std() < 1e-9:
            return 0.0
        Rs = R - R.max()  # shift for numerical stability (softmax-invariant)
        B = len(R)
        if self.beta_rule == "fixed":
            return self.beta / (R.std() + 1e-8)
        if self.beta_rule == "ess":  # ESS shrinks as beta grows
            ess = lambda w: 1.0 / np.sum(w * w)
            return self._bisect_beta(Rs, ess, self.target_ess * B, increasing=False)
        # kl: KL(tilted || sampling) = log B - H(weights), grows as beta grows
        kl = lambda w: np.log(B) + np.sum(w * np.log(w + 1e-12))
        target = min(self.target_kl, 0.999 * np.log(B))
        return self._bisect_beta(Rs, kl, target, increasing=True)

    def _weights(self, R: np.ndarray) -> np.ndarray:
        if self.mode == "quantile":
            q = np.quantile(R, 1.0 - self.epsilon)
            return np.where(R >= q, R - q, 0.0)  # train only on the top tail
        # entropic exponential tilt, centered to sum ~0
        b = self._entropic_beta(R)
        self._last_beta = float(b)
        logits = b * (R - R.max())               # subtract max for stability
        sm = np.exp(logits)
        sm /= sm.sum()
        return sm * len(R) - 1.0                 # O(1) scale, mean 0

    def tell(self, candidates, results: list[Result]) -> None:
        R = np.array([r.reward for r in results])
        self.policy.reinforce(self._weights(R), self.ent_coef)

    def diagnostics(self) -> dict:
        return {"policy_entropy": self.policy.last_entropy(),
                "beta": self._last_beta}
