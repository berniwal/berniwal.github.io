"""Sanity checks for the shared seams: grammar round-trips, the exact targets
score ~1.0 and pass the success check, and the verifier counts calls.

Run: python -m tests.test_core   (from the repo root, src on PYTHONPATH)
"""
from __future__ import annotations

import numpy as np

from sia.expression import (GRAMMAR, Node, complexity, evaluate, from_prefix,
                            leaf, parse_expression, random_tree, to_prefix)
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


def test_arbitrary_constant_eval():
    # constants outside the grammar set must still evaluate (Layer 1 LLM output)
    x = np.linspace(-2, 2, 7)
    node = Node("+", [Node("*", [leaf("3.0"), leaf("x")]), leaf("0.25")])
    assert np.allclose(evaluate(node, x), 3.0 * x + 0.25)


def test_parse_expression_matches_eval():
    x = np.linspace(-3, 3, 11)
    cases = {
        "x*x + sin(x)": x * x + np.sin(x),
        "x**2 + sin(x)": x * x + np.sin(x),       # power expansion
        "3*x + 0.5": 3 * x + 0.5,                  # arbitrary constants
        "cos(2*x) - x": np.cos(2 * x) - x,
        "-x + 1": -x + 1,                          # unary minus
    }
    for text, expected in cases.items():
        node = parse_expression(text)
        assert node is not None, text
        assert np.allclose(evaluate(node, x), expected), text


def test_parse_expression_rejects_out_of_grammar():
    for bad in ["sqrt(x)", "x**2.5", "exp(x)", "x +", "tan(x)", "import os"]:
        assert parse_expression(bad) is None, bad


def test_parse_expression_strips_prose():
    node = parse_expression("Sure! The expression is:\n```\ny = x*x + sin(x)\n```")
    assert node is not None
    x = np.linspace(-2, 2, 5)
    assert np.allclose(evaluate(node, x), x * x + np.sin(x))


def test_verifier_counts_calls():
    task = make_task("medium", seed=0)
    ver = Verifier(task)
    rng = np.random.default_rng(0)
    for _ in range(100):
        ver(random_tree(GRAMMAR, rng))
    assert ver.calls == 100


def test_nrmse_reward_mode():
    task = make_task("medium", seed=0)
    v = Verifier(task, length_penalty=0.0, reward_mode="nrmse")
    # exact target -> NRMSE 0 -> reward ~1
    assert v(task.target_expr).reward > 0.999
    # predicting the mean -> RMSE = std(y) -> NRMSE = 1 -> reward exactly 0.5
    mean_node = leaf(repr(float(np.mean(task.y_train))))
    assert abs(v(mean_node).reward - 0.5) < 1e-9, v(mean_node).reward
    # bad mode rejected
    try:
        Verifier(task, reward_mode="bogus")
        assert False, "should have raised"
    except ValueError:
        pass


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
