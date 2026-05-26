#!/usr/bin/env python3
"""Export a compact, pre-computed replay of Layer 0 runs to JSON, so a static web
page (the blog) can scrub the training dynamics batch-by-batch without running any
Python.

This mirrors the Streamlit step-through, reusing the SAME ``app_engine`` primitives
(``build_layer0_state`` / ``step`` / ``batch_summary``) so the live tool and the baked
replay cannot drift. It is ONE representative seed per method -- a step-through is a
single run -- while the statistical claim (mean over 20 seeds) lives in the committed
``scaling.png``. Each checkpoint records what the visualizer shows: best-so-far reward,
batch diversity / policy entropy / success fraction, the top-k batch proposals (the
collapse made legible), and the current best expression evaluated on a fixed x-grid
(so the page can draw the fit overlay without an expression evaluator in JS).

    python export_replay.py                          # all targets, mse + nrmse
    python export_replay.py --target medium --reward mse --budget 100000

Writes results/layer0_<reward>/replay.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import app_engine as eng  # also bootstraps sys.path so `import sia` works
from sia.expression import evaluate, to_infix
from sia.plotting import STYLE

ROOT = Path(__file__).resolve().parent


def _round(values, n: int = 4):
    """Round a 1-D array for JSON, mapping non-finite entries to null."""
    out = []
    for v in values:
        v = float(v)
        out.append(round(v, n) if math.isfinite(v) else None)
    return out


def _scalar(v, n: int = 4):
    v = float(v)
    return round(v, n) if math.isfinite(v) else None


def _checkpoint_batches(n_batches: int, n_checkpoints: int, spacing: str) -> set[int]:
    """Pick which batch indices (1-based) to snapshot.

    `log` spacing puts most frames early, where the greedy collapse and the initial
    climb happen, while still reaching the end of a long (e.g. 2M-call) run -- so a
    single replay shows BOTH the early collapse and the late recovery without bloating
    the frame count. `linear` is the old uniform spacing.
    """
    if n_checkpoints >= n_batches:
        return set(range(1, n_batches + 1))
    if spacing == "log":
        idx = np.geomspace(1, n_batches, n_checkpoints)
    else:
        idx = np.linspace(1, n_batches, n_checkpoints)
    pts = set(int(round(v)) for v in idx)
    pts.add(n_batches)  # always include the final frame
    return pts


def export_target(target: str, reward_mode: str, seed: int, budget: int,
                  batch_size: int, n_checkpoints: int, k_rows: int,
                  n_grid: int, spacing: str = "log") -> dict:
    """Run every Layer-0 method once on `target` and snapshot ~n_checkpoints frames."""
    task = eng.make_task(target, seed=seed)
    xs = np.linspace(float(task.x_train.min()), float(task.x_train.max()), n_grid)
    with np.errstate(all="ignore"):
        y_target = evaluate(task.target_expr, xs)

    n_batches = budget // batch_size
    record_at = _checkpoint_batches(n_batches, n_checkpoints, spacing)

    methods: dict = {}
    for key in eng.LAYER0_PRESETS:
        state = eng.build_layer0_state(key, task, seed, batch_size, reward_mode)
        checkpoints = []
        for b in range(n_batches):
            eng.step(state, 1)
            if (b + 1) in record_at:
                y_pred = None
                if state.best_expr is not None:
                    with np.errstate(all="ignore"):
                        y_pred = _round(evaluate(state.best_expr, xs))
                rows = [[e, n, _scalar(r)]
                        for e, n, r in eng.batch_summary(state, k=k_rows)]
                checkpoints.append({
                    "calls": int(state.calls[-1]),
                    "best": _scalar(state.best_reward[-1]),
                    "best_infix": to_infix(state.best_expr)
                    if state.best_expr is not None else "",
                    "y_pred": y_pred,
                    "diversity": _scalar(state.diversity[-1]),
                    "entropy": _scalar(state.entropy[-1]),
                    "success_frac": _scalar(state.success_frac[-1]),
                    "rows": rows,
                })
        methods[key] = {"label": STYLE[key][1], "color": STYLE[key][0],
                        "checkpoints": checkpoints}

    return {
        "target_infix": to_infix(task.target_expr),
        "x_grid": _round(xs),
        "y_target": _round(y_target),
        "methods": methods,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all",
                    choices=["all", "easy", "medium", "harder"])
    ap.add_argument("--reward", default="all", choices=["all", "mse", "nrmse"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=100000,
                    help="replay budget in verifier calls. The JSON size depends on "
                         "--checkpoints, NOT on the budget, so a 2M replay is the same "
                         "file size as a 100k one -- only the generation time scales. "
                         "Use --budget 2000000 --spacing log to show the full recovery.")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--checkpoints", type=int, default=64)
    ap.add_argument("--spacing", default="log", choices=["log", "linear"],
                    help="log = dense frames early (keeps the collapse resolution while "
                         "still reaching the end of a long run); linear = uniform")
    ap.add_argument("--rows", type=int, default=6, help="top-k batch proposals per frame")
    ap.add_argument("--grid", type=int, default=60, help="x-points for the fit overlay")
    args = ap.parse_args()

    targets = ["easy", "medium", "harder"] if args.target == "all" else [args.target]
    rewards = ["mse", "nrmse"] if args.reward == "all" else [args.reward]

    for rm in rewards:
        out_dir = ROOT / "results" / f"layer0_{rm}"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "reward_mode": rm, "seed": args.seed, "budget": args.budget,
            "batch_size": args.batch_size, "targets": {},
        }
        for t in targets:
            payload["targets"][t] = export_target(
                t, rm, seed=args.seed, budget=args.budget,
                batch_size=args.batch_size, n_checkpoints=args.checkpoints,
                k_rows=args.rows, n_grid=args.grid, spacing=args.spacing)
        path = out_dir / "replay.json"
        # allow_nan=False: fail loudly if any non-finite slipped through sanitizing,
        # since NaN/Infinity are invalid JSON and would break the web reader.
        path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        kb = path.stat().st_size / 1024
        print(f"wrote {path}  ({kb:.0f} KB, "
              f"{len(targets)} targets x {len(eng.LAYER0_PRESETS)} methods)")


if __name__ == "__main__":
    main()
