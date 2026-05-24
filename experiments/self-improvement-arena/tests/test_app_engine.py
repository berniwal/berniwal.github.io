"""Tests for the Streamlit engine (app_engine.py) -- the steppable core, WITHOUT
Streamlit or MLX. Exercises the Layer 0 build/step/record loop, the budget
accounting (1 step == batch_size verifier calls), and the plot-data helpers.

Run: PYTHONPATH=src:. python -m tests.test_app_engine
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import app_engine as eng  # noqa: E402


def test_presets_all_build_and_step():
    task = eng.make_task("easy", seed=0)
    for key in eng.LAYER0_PRESETS:
        s = eng.build_layer0_state(key, task, seed=0, batch_size=20)
        eng.step(s, 2)
        assert s.steps == 2
        assert s.verifier.calls == 2 * 20            # budget == batch_size per step
        assert len(s.best_reward) == 2 and len(s.diversity) == 2
        assert len(s.success_frac) == 2
        assert 0.0 <= s.diversity[-1] <= 1.0
        assert 0.0 <= s.success_frac[-1] <= 1.0


def test_step_accumulates_budget_and_history():
    task = eng.make_task("medium", seed=1)
    s = eng.build_layer0_state("random", task, seed=1, batch_size=50)
    eng.step(s, 3)
    eng.step(s, 1)
    assert s.steps == 4
    assert s.verifier.calls == 4 * 50
    assert s.calls == [50, 100, 150, 200]            # cumulative, monotone
    # best-so-far reward is non-decreasing by construction
    assert all(b2 >= b1 for b1, b2 in zip(s.best_reward, s.best_reward[1:]))


def test_best_expr_tracks_best_reward():
    task = eng.make_task("easy", seed=2)
    s = eng.build_layer0_state("gp", task, seed=2, batch_size=100)
    eng.step(s, 5)
    assert s.best_expr is not None
    res = s.verifier(s.best_expr)
    assert abs(res.reward - s.best) < 1e-9           # the stored node really is the best


def test_rl_arms_report_entropy_random_does_not():
    task = eng.make_task("easy", seed=0)
    g = eng.build_layer0_state("greedy", task, seed=0, batch_size=50)
    r = eng.build_layer0_state("random", task, seed=0, batch_size=50)
    eng.step(g, 1); eng.step(r, 1)
    assert np.isfinite(g.entropy[-1])                # RNN policy reports entropy
    assert np.isnan(r.entropy[-1])                   # random search has none


def test_fit_overlay_shapes_and_target():
    task = eng.make_task("medium", seed=0)
    s = eng.build_layer0_state("random", task, seed=0, batch_size=30)
    eng.step(s, 1)
    xs, y_target, y_pred = eng.fit_overlay(task, s.best_expr, n=128)
    assert xs.shape == (128,) and y_target.shape == (128,)
    # target curve equals the true target function evaluated on the grid
    from sia.expression import evaluate
    assert np.allclose(y_target, evaluate(task.target_expr, xs))
    assert y_pred is None or y_pred.shape == (128,)
    # with no expression yet, prediction is None
    _, _, none_pred = eng.fit_overlay(task, None)
    assert none_pred is None


def test_batch_summary_counts_sum_to_batch():
    task = eng.make_task("easy", seed=0)
    s = eng.build_layer0_state("greedy", task, seed=0, batch_size=40)
    eng.step(s, 1)
    rows = eng.batch_summary(s, k=100)               # k large -> all distinct exprs
    assert sum(n for _, n, _ in rows) == 40          # counts partition the batch
    assert rows == sorted(rows, key=lambda t: t[1], reverse=True)  # most-frequent first


def test_capability_detection_is_boolean():
    assert isinstance(eng.mlx_available(), bool)     # never raises off-Mac
    assert "/" in eng.platform_label()


def test_layer1_arm_validation():
    task = eng.make_task("easy", seed=0)
    try:
        eng.build_layer1_state("nope", task, 0, model=None, tokenizer=None)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
