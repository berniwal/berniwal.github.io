#!/usr/bin/env python3
"""The constraints x reward ablation: 4 runs that isolate the ONE thing that turns
greedy RL from working into collapsing, holding DSR's tuned lr/batch/entropy/risk
fixed throughout (so it is not confounded by hyperparameter tuning):

    MSE   + constraints   ->  greedy works
    MSE   - constraints   ->  greedy partial
    NRMSE + constraints   ->  greedy works
    NRMSE - constraints   ->  greedy collapses

Risk-seeking is included in every run as the control: it is expected to solve
regardless of constraints (it does not need the guardrails).

    python run_ablation.py --budget 150000 --seeds 5     # smoke + timing
    python run_ablation.py --budget 2000000 --seeds 20   # full

Writes results/abl_<reward>_<con|nocon>/{curves,diversity,scaling}.png + results.md.
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

from sia.metrics import success_rate, median_evals_to_solve  # noqa: E402
from sia.plotting import make_all  # noqa: E402
from sia.runner import run_experiment  # noqa: E402

# DSR Table 3 hyperparameters, shared by every RNN arm so the only difference
# between greedy / risk / entropic stays the objective weighting.
def _rnn(constraints: bool, **extra) -> dict:
    return dict(batch_size=1000, hidden=32, max_length=30, min_length=4,
                constraints=constraints, lr=0.0005, ent_coef=0.005, **extra)


def make_config(reward: str, constraints: bool, budget: int, seeds: int) -> dict:
    return dict(
        budget=budget, seeds=seeds, n_points=30, x_range=[-3.0, 3.0],
        length_penalty=0.001, eps_success=1.0e-6, reward_mode=reward, log_every=5,
        targets=["easy", "medium", "harder"],
        methods={
            "random": dict(proposer="random", batch_size=200, max_depth=4),
            "gp": dict(proposer="gp", pop_size=200, batch_size=200, tournament_size=5,
                       crossover_rate=0.6, mutation_rate=0.3, immigrant_rate=0.1,
                       max_depth=4, max_complexity=30),
            "greedy": dict(proposer="greedy", **_rnn(constraints)),
            "cvar": dict(proposer="risk", **_rnn(constraints, mode="cvar", epsilon=0.05)),
            "risk": dict(proposer="risk", **_rnn(constraints, mode="quantile", epsilon=0.05)),
            "risk_entropic": dict(proposer="risk",
                                  **_rnn(constraints, mode="entropic",
                                         beta_rule="fixed", beta=2.0)),
        })


def _cell(logs, target, method):
    runs = [lg for lg in logs if lg.target == target and lg.method == method]
    sr = success_rate(runs)
    med = median_evals_to_solve(runs)
    return sr, (int(med) if med is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing logs and recompute (default: resume — "
                         "re-running after a crash continues where it left off)")
    args = ap.parse_args()

    setups = [("mse", True), ("mse", False), ("nrmse", True), ("nrmse", False)]
    summary = []
    for reward, con in setups:
        tag = f"{reward}_{'con' if con else 'nocon'}"
        out = ROOT / "results" / f"abl_{tag}"
        cfg = make_config(reward, con, args.budget, args.seeds)
        print(f"\n>>> {tag}: {len(cfg['methods'])}x3 targets x{args.seeds} seeds, "
              f"budget={args.budget}, workers={args.workers}")
        t0 = time.time()
        logs = run_experiment(cfg, log_dir=out / "logs", resume=not args.fresh,
                              workers=args.workers)
        dt = time.time() - t0
        make_all(logs, out, reward)
        g_sr, g_med = _cell(logs, "medium", "greedy")
        r_sr, r_med = _cell(logs, "medium", "risk")
        summary.append((tag, dt, g_sr, g_med, r_sr, r_med))
        print(f"    done in {dt:.0f}s  ->  {out}")

    print("\n==================  ablation summary (target=medium)  ==================")
    print(f"{'setup':14s} {'wall':>7s} {'greedy_succ':>12s} {'greedy_med':>11s} "
          f"{'risk_succ':>10s} {'risk_med':>9s}")
    for tag, dt, g_sr, g_med, r_sr, r_med in summary:
        print(f"{tag:14s} {dt:>6.0f}s {g_sr:>12.2f} {str(g_med):>11s} "
              f"{r_sr:>10.2f} {str(r_med):>9s}")

    full_runs = 6 * 3 * 20
    this_runs = 6 * 3 * args.seeds
    factor = (2_000_000 / args.budget) * (full_runs / this_runs)
    tot = sum(s[1] for s in summary)
    print(f"\nFull-run (2M, 20 seeds) estimate per setup ~ {factor:.0f}x this smoke.")
    print(f"This smoke total: {tot:.0f}s ({tot/60:.1f} min) for 4 setups.")
    print(f"=> full 4-setup estimate ~ {factor*tot/3600:.1f} h "
          f"(~{factor*tot/4/3600:.1f} h per setup).")


if __name__ == "__main__":
    main()
