#!/usr/bin/env python3
"""The entropy ablation: does HIERARCHICAL entropy (Landajuela et al. 2021) un-collapse
greedy RL on its own -- a second, independent anti-collapse lever to the constraints
shown in run_ablation.py? Constraints are held OFF throughout (the collapse regime),
DSR lr/batch fixed, and we vary only the entropy bonus x reward:

    flat entropy  (gamma=1.0,  ent_coef=0.005)  -- original DSR / our prior behavior
    hierarchical  (gamma=0.85, ent_coef=0.02)   -- early-token-weighted (Landajuela 2021)

Expectation: greedy collapses with flat entropy (esp. under NRMSE) and is (partly?)
rescued by hierarchical entropy; risk-seeking solves either way (the control).

    python run_entropy.py --budget 2000000 --seeds 20      # full (RunPod)
    python run_entropy.py --budget 150000 --seeds 5        # smoke

Writes results/ent_<reward>_<flat|hier>/{curves,diversity,scaling}.png + results.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SRC = str(ROOT / "src")
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from sia.metrics import median_evals_to_solve, success_rate  # noqa: E402
from sia.plotting import make_all  # noqa: E402
from sia.runner import run_experiment  # noqa: E402

# DSR lr/batch, constraints OFF; only the entropy bonus differs between setups.
def _rnn(ent_coef: float, gamma: float, **extra) -> dict:
    return dict(batch_size=1000, hidden=32, max_length=30, min_length=4,
                constraints=False, lr=0.0005, ent_coef=ent_coef,
                entropy_gamma=gamma, **extra)


def make_config(reward: str, hierarchical: bool, budget: int, seeds: int) -> dict:
    ec, g = (0.02, 0.85) if hierarchical else (0.005, 1.0)
    return dict(
        budget=budget, seeds=seeds, n_points=30, x_range=[-3.0, 3.0],
        length_penalty=0.001, eps_success=1.0e-6, reward_mode=reward, log_every=5,
        targets=["easy", "medium", "harder"],
        methods={
            "random": dict(proposer="random", batch_size=200, max_depth=4),
            "gp": dict(proposer="gp", pop_size=200, batch_size=200, tournament_size=5,
                       crossover_rate=0.6, mutation_rate=0.3, immigrant_rate=0.1,
                       max_depth=4, max_complexity=30),
            "greedy": dict(proposer="greedy", **_rnn(ec, g)),
            "cvar": dict(proposer="risk", **_rnn(ec, g, mode="cvar", epsilon=0.05)),
            "risk": dict(proposer="risk", **_rnn(ec, g, mode="quantile", epsilon=0.05)),
            "risk_entropic": dict(proposer="risk",
                                  **_rnn(ec, g, mode="entropic", beta_rule="fixed", beta=2.0)),
        })


def _cell(logs, target, method):
    runs = [lg for lg in logs if lg.target == target and lg.method == method]
    med = median_evals_to_solve(runs)
    return success_rate(runs), (int(med) if med is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing logs (default: resume)")
    args = ap.parse_args()

    summary = []
    for reward in ("mse", "nrmse"):
        for hier in (False, True):
            tag = f"{reward}_{'hier' if hier else 'flat'}"
            out = ROOT / "results" / f"ent_{tag}"
            cfg = make_config(reward, hier, args.budget, args.seeds)
            print(f"\n>>> {tag}: 6 methods x3 targets x{args.seeds} seeds, "
                  f"budget={args.budget}, workers={args.workers}")
            t0 = time.time()
            logs = run_experiment(cfg, log_dir=out / "logs", resume=not args.fresh,
                                  workers=args.workers)
            dt = time.time() - t0
            make_all(logs, out, reward)
            g_sr, g_med = _cell(logs, "medium", "greedy")
            r_sr, r_med = _cell(logs, "medium", "risk")
            summary.append((tag, dt, g_sr, g_med, r_sr, r_med))
            print(f"    done in {dt:.0f}s -> {out}")

    print("\n==============  entropy ablation summary (target=medium)  ==============")
    print(f"{'setup':14s} {'wall':>7s} {'greedy_succ':>12s} {'greedy_med':>11s} "
          f"{'risk_succ':>10s} {'risk_med':>9s}")
    for tag, dt, g_sr, g_med, r_sr, r_med in summary:
        print(f"{tag:14s} {dt:>6.0f}s {g_sr:>12.2f} {str(g_med):>11s} "
              f"{r_sr:>10.2f} {str(r_med):>9s}")


if __name__ == "__main__":
    main()
