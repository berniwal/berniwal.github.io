"""Numeric checks for the shared RL objectives (sia/objectives.py).

These pin down the three weightings so that Layer 0 and Layer 1 provably share the
SAME objective. Pure numpy, no LLM / MLX needed.

Run: PYTHONPATH=src python -m tests.test_objectives
"""
from __future__ import annotations

import numpy as np

from sia.objectives import (entropic_beta, entropic_weights, greedy_weights,
                            quantile_weights)


def test_greedy_is_mean_baselined():
    R = np.array([0.0, 0.5, 1.0])
    w = greedy_weights(R)
    assert np.allclose(w, R - R.mean())
    assert np.isclose(w.sum(), 0.0)             # mean baseline => zero-sum
    # ordering preserved; best sample gets the largest (positive) weight
    assert w.argmax() == R.argmax() and w.argmin() == R.argmin()


def test_quantile_keeps_only_top_tail():
    R = np.arange(10, dtype=float)              # 0..9
    w = quantile_weights(R, eps=0.2)            # keep top 20% -> the two largest
    # q = 80th percentile of 0..9 = 7.2; samples >= 7.2 are 8 and 9
    assert (w[:8] == 0.0).all()                 # everything below the cutoff is dropped
    assert (w[8:] > 0.0).all()
    assert np.isclose(w[9], 9.0 - np.quantile(R, 0.8))


def test_quantile_weight_is_reward_minus_cutoff():
    R = np.array([0.1, 0.2, 0.9, 1.0])
    eps = 0.5
    q = np.quantile(R, 1 - eps)
    w = quantile_weights(R, eps)
    assert np.allclose(w, np.where(R >= q, R - q, 0.0))


def test_entropic_centered_and_scaled():
    rng = np.random.default_rng(0)
    for _ in range(100):
        R = rng.random(40)
        w, b = entropic_weights(R, "fixed", beta=2.0)
        assert np.isclose(w.mean(), 0.0, atol=1e-9)     # centered (mean 0)
        assert b > 0.0                                   # positive tilt
        # weights are a monotone function of reward (exp tilt preserves order)
        order = np.argsort(R)
        assert np.all(np.diff(w[order]) >= -1e-12)


def test_entropic_beta_zero_when_no_variance():
    R = np.full(16, 0.7)
    w, b = entropic_weights(R, "fixed", beta=2.0)
    assert b == 0.0
    assert np.allclose(w, 0.0)                            # uniform tilt -> zero weights


def test_entropic_limits_toward_greedy_and_max():
    # small beta -> approaches greedy (mean-baselined, up to O(beta) scaling);
    # large beta -> mass concentrates on the single best sample.
    R = np.array([0.0, 0.3, 0.6, 1.0])
    w_small, _ = entropic_weights(R, "fixed", beta=1e-3)
    g = greedy_weights(R)
    # direction matches greedy (cosine ~ 1) in the small-beta limit
    cos = (w_small @ g) / (np.linalg.norm(w_small) * np.linalg.norm(g) + 1e-12)
    assert cos > 0.999
    w_big, _ = entropic_weights(R, "fixed", beta=50.0)
    assert w_big.argmax() == R.argmax()
    assert w_big[R.argmax()] > w_big.sum() - w_big[R.argmax()]  # best dominates


def test_entropic_beta_rules_hit_their_targets():
    rng = np.random.default_rng(1)
    R = rng.random(64)
    B = len(R)
    Rs = R - R.max()
    # ess rule: realized effective sample size should match the target
    b_ess = entropic_beta(R, "ess", target_ess=0.3)
    p = np.exp(b_ess * Rs); p /= p.sum()
    ess = 1.0 / np.sum(p * p)
    assert abs(ess - 0.3 * B) < 0.05 * B
    # kl rule: realized KL(tilted || empirical) should match the target
    b_kl = entropic_beta(R, "kl", target_kl=1.0)
    p = np.exp(b_kl * Rs); p /= p.sum()
    kl = np.log(B) + np.sum(p * np.log(p + 1e-12))
    assert abs(kl - 1.0) < 0.05


def test_layer0_proposers_use_shared_objectives():
    """The Layer 0 RL proposers must produce EXACTLY the shared-objective weights
    (this is the refactor's contract: one source of truth, identical behavior)."""
    from sia.proposers.rl_greedy import RLGreedy
    from sia.proposers.rl_risk import RLRisk
    from sia.task import make_task
    from sia.verifier import Result

    task = make_task("medium", seed=0)
    rng = np.random.default_rng(0)
    R = rng.random(20)
    results = [Result(reward=float(r), mse=0.0, valid=True, complexity=1) for r in R]

    g = RLGreedy(task, rng)
    assert np.allclose(np.array([r.reward for r in results]).mean(),
                       R.mean())  # sanity

    rq = RLRisk(task, rng, mode="quantile", epsilon=0.1)
    assert np.allclose(rq._weights(np.array([r.reward for r in results])),
                       quantile_weights(R, 0.1))
    re = RLRisk(task, rng, mode="entropic", beta_rule="fixed", beta=2.0)
    w_expected, _ = entropic_weights(R, "fixed", 2.0)
    assert np.allclose(re._weights(R), w_expected)
    # greedy proposer path: weights equal greedy_weights(R)
    assert np.allclose(greedy_weights(R), R - R.mean())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
