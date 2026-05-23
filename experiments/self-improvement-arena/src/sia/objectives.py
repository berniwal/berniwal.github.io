"""The RL objectives, as pure reward -> per-sample weight functions.

A policy-gradient / weighted-SFT update is ALWAYS

    maximize   sum_i  w_i * sum_t log pi(token_t^i)        (+ optional entropy bonus)

and the arms differ ONLY in how the scalar rewards ``R`` become the per-sample
weight vector ``w`` -- a whole risk spectrum from one gradient:

    cvar_weights      risk-AVERSE : lift the worst eps-tail        (Tamar 2014; EPOpt 2016)
    greedy_weights    neutral     : E[R], the mean-baselined sample
    entropic_weights  soft seeking: e^{beta R} tilt               (Jiang 2025 / TTT-Discover)
    quantile_weights  risk-SEEKING: chase the best (1-eps)-tail   (DSR, Petersen 2021)

The two tails (cvar vs. quantile) are exact mirrors: same conditional-tail-
expectation gradient, opposite tail and opposite intent. Those formulas live here,
once, so that BOTH

  * Layer 0 (numpy RNN policy: ``proposers/rl_greedy.py``, ``proposers/rl_risk.py``)
  * Layer 1 (LLM + LoRA: ``layer1/lora_proposer.py``)

share a single source of truth. The backprop differs between layers (manual BPTT
vs. MLX autograd), but the *objective* — the thing that defines greedy vs.
risk-seeking vs. entropic — is identical, and is exactly this module.

Everything here is plain numpy (no heavy deps), so Layer 0 stays numpy-only and
these functions are unit-testable without an LLM or Apple Silicon.

See the README "three RL objectives" table and lineage note for the theory; the
docstrings below give the one-line intuition per objective.
"""
from __future__ import annotations

import numpy as np


def greedy_weights(R: np.ndarray) -> np.ndarray:
    """Greedy / expected-reward REINFORCE: ``w_i = R_i - mean(R)``.

    Maximizes ``E[R]`` (the *average* sample) with a mean baseline for variance
    reduction. This is the arm that mode-collapses: optimizing the typical sample
    rewards a simple-but-wrong attractor over the rare exact hit.
    """
    R = np.asarray(R, dtype=float)
    return R - R.mean()


def quantile_weights(R: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Risk-seeking quantile gradient (Deep Symbolic Regression, Petersen 2021).

    Keep only the top-``eps`` of the batch and weight each kept sample by
    ``R_i - q``, where ``q`` is the ``(1 - eps)`` reward quantile; everything
    below ``q`` gets weight 0. The quantile *is* the baseline. This reinforces a
    diverse *set* of good expressions instead of one mode, so it resists collapse.

    DSR obtains this by *inverting* the risk-averse CVaR policy gradient
    (``cvar_weights`` below) from the lower tail to the upper tail -- same
    conditional-tail-expectation machinery, opposite tail and opposite intent.
    """
    R = np.asarray(R, dtype=float)
    q = np.quantile(R, 1.0 - eps)
    return np.where(R >= q, R - q, 0.0)  # train only on the top tail


def cvar_weights(R: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Risk-AVERSE CVaR policy gradient -- the exact MIRROR of ``quantile_weights``.

    Conditional Value-at-Risk optimizes the *worst* eps-tail: keep only the bottom
    ``eps`` of the batch and weight each by ``R_i - q``, where ``q`` is now the
    ``eps`` quantile (the lower-tail VaR). The kept weights are ``<= 0``, so
    REINFORCE pushes probability *away* from the catastrophic tail -- it lifts the
    worst case rather than chasing the best.

    This is the classic risk-averse objective of the CVaR policy-gradient line
    (Tamar, Glassner & Mannor 2014, "Policy Gradients Beyond Expectations: CVaR";
    Rajeswaran et al. 2016, EPOpt, which applies it across a model ensemble for
    robustness). DSR's risk-seeking ``quantile_weights`` is literally this gradient
    inverted to the upper tail. For *discovery* -- where you want the single exact
    hit, not a safe floor -- this is the deliberately "wrong direction" baseline
    that anchors the risk-averse end of the spectrum (cvar -> greedy E[R] ->
    entropic J_beta -> risk-seeking quantile).
    """
    R = np.asarray(R, dtype=float)
    q = np.quantile(R, eps)
    return np.where(R <= q, R - q, 0.0)  # train only on the worst tail (weights <= 0)


def _bisect_beta(Rs: np.ndarray, stat, target: float, increasing: bool) -> float:
    """Find ``beta >= 0`` with ``stat(softmax(beta * Rs)) == target``, assuming
    ``stat`` is monotone in beta. ``increasing`` says whether ``stat`` grows with
    beta (KL) or shrinks (ESS)."""
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


def entropic_beta(R: np.ndarray, beta_rule: str = "fixed", beta: float = 2.0,
                  target_ess: float = 0.3, target_kl: float = 1.0) -> float:
    """Choose the entropic temperature ``beta`` for one batch.

    - ``fixed``: ``beta / std(R)`` -- constant tilt strength, scale-normalized
      (Jiang et al. 2025 use a constant beta).
    - ``ess``: pick beta so the exponential weights have a target effective
      sample size (a standard importance-sampling self-tuning rule).
    - ``kl``: pick beta so the induced (reward-tilted) distribution sits a target
      KL away from the batch sampling distribution. Our reading of TTT-Discover's
      adaptive beta; we constrain ``KL(tilted || empirical) = log B - H(weights)``,
      which is batch-computable and lr-independent (their exact functional may differ).
    """
    R = np.asarray(R, dtype=float)
    if R.std() < 1e-9:
        return 0.0
    Rs = R - R.max()  # shift for numerical stability (softmax-invariant)
    B = len(R)
    if beta_rule == "fixed":
        return beta / (R.std() + 1e-8)
    if beta_rule == "ess":  # ESS shrinks as beta grows
        ess = lambda w: 1.0 / np.sum(w * w)
        return _bisect_beta(Rs, ess, target_ess * B, increasing=False)
    if beta_rule == "kl":  # KL(tilted || sampling) = log B - H(weights), grows with beta
        kl = lambda w: np.log(B) + np.sum(w * np.log(w + 1e-12))
        target = min(target_kl, 0.999 * np.log(B))
        return _bisect_beta(Rs, kl, target, increasing=True)
    raise ValueError(f"beta_rule must be fixed/ess/kl, got {beta_rule!r}")


def entropic_weights(R: np.ndarray, beta_rule: str = "fixed", beta: float = 2.0,
                     target_ess: float = 0.3, target_kl: float = 1.0
                     ) -> tuple[np.ndarray, float]:
    """Entropic exponential tilt (Jiang et al. 2025 / TTT-Discover): weights
    proportional to ``e^{beta R}``, centered to mean ~0 with O(1) scale.

    Maximizes ``J_beta = (1/beta) log E[e^{beta R}]``; ``beta -> 0`` recovers
    greedy, ``beta -> inf`` recovers max. Returns ``(weights, beta)`` -- the
    chosen beta is surfaced for diagnostics.
    """
    R = np.asarray(R, dtype=float)
    b = entropic_beta(R, beta_rule, beta, target_ess, target_kl)
    logits = b * (R - R.max())  # subtract max for stability
    sm = np.exp(logits)
    sm /= sm.sum()
    return sm * len(R) - 1.0, float(b)  # O(1) scale, mean 0
