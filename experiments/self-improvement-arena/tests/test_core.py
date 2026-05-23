"""Sanity checks for the shared seams: grammar round-trips, the exact targets
score ~1.0 and pass the success check, and the verifier counts calls.

Run: python -m tests.test_core   (from the repo root, src on PYTHONPATH)
"""
from __future__ import annotations

import numpy as np

from sia.expression import (GRAMMAR, evaluate, from_prefix, random_tree,
                            to_prefix)
from sia.task import TARGET_EXPRS, make_task
from sia.verifier import Verifier


def test_prefix_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        t = random_tree(GRAMMAR, rng, max_depth=5)
        assert from_prefix(to_prefix(t)) == t


def test_from_prefix_rejects_malformed():
    assert from_prefix(["+", "x"]) is None          # too short
    assert from_prefix(["x", "x"]) is None           # leftover token
    assert from_prefix(["sin"]) is None              # missing operand


def test_targets_reachable_and_perfect():
    for name, expr in TARGET_EXPRS.items():
        task = make_task(name, seed=0)
        ver = Verifier(task, length_penalty=0.001)
        res = ver(expr)
        # exact target: tiny MSE -> reward ~ 1 minus a small length penalty
        assert res.valid and res.mse < 1e-12, (name, res)
        assert res.reward > 0.95, (name, res.reward)
        assert ver.success(expr), name


def test_targets_match_numpy_fns():
    for name in TARGET_EXPRS:
        task = make_task(name, seed=1)
        y = evaluate(TARGET_EXPRS[name], task.x_train)
        assert np.allclose(y, task.y_train, atol=1e-9), name


def test_verifier_counts_calls():
    task = make_task("medium", seed=0)
    ver = Verifier(task)
    rng = np.random.default_rng(0)
    for _ in range(100):
        ver(random_tree(GRAMMAR, rng))
    assert ver.calls == 100


def test_invalid_expression_scores_zero():
    from sia.expression import Node, leaf
    task = make_task("easy", seed=0)
    ver = Verifier(task)
    bad = Node("/", [leaf("x"), Node("-", [leaf("x"), leaf("x")])])  # x/(x-x) -> inf
    res = ver(bad)
    assert not res.valid and res.reward == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
