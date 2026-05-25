#!/usr/bin/env python3
"""Does our greedy (vanilla PG = DSR's VPG) reproduce DSR's collapse on Nguyen?

Under DSR's published hyperparameters, the paper reports VPG recovery of 96/47/4/1%
on Nguyen-1..4 and DSR (risk-seeking) 100/100/100/100% -- the headline "greedy
collapses, risk recovers" result. In our sandbox, with the SAME faithful HPs
(entropy_weight=0.03, entropy_gamma=0.7, inv_nrmse reward, NO parsimony penalty,
batch 1000, 2M evaluations, lr 5e-4, epsilon 0.05), local runs at max_length=30
recovered nguyen-3/4 at ~100% for BOTH arms -- i.e. greedy did NOT collapse.

The leading suspect is search-space size: DSR's repo uses max_length=64, we used 30.
This script sweeps max_length in {30, 64} for both arms so we can see whether the
larger space alone reproduces VPG's collapse (greedy drops, risk stays high). Uses
DSR's strict recovery criterion (exact symbolic equivalence via SymPy); reports both
"anytime" (any sampled candidate was exact) and "final-best" (the returned best-
reward expression is exact -- DSR's actual definition).

    python3 run_nguyen_collapse.py --seeds 25 --max-lengths 30,64

Writes per-run JSON to results/nguyen-collapse/maxlen<L>/logs/ (resumable) and
prints a recovery table.
"""
from __future__ import annotations

import argparse
import os

from sia.expression import parse_expression, sympy_equivalent
from sia.runner import run_experiment
from sia.task import NGUYEN_SYMPY

# DSR-faithful policy/training hyperparameters (dso config_regression.json), shared
# by both arms -- ONLY the objective differs (greedy = vanilla PG; risk = top-eps).
FAITHFUL = dict(batch_size=1000, hidden=32, min_length=4, constraints=True,
                lr=0.0005, ent_coef=0.03, entropy_gamma=0.7)


def recovery_table(logs, targets, seeds):
    rows = []
    for method in ("greedy", "risk"):
        for t in targets:
            runs = [lg for lg in logs if lg.method == method and lg.target == t]
            anytime = sum(lg.success_symbolic for lg in runs)
            final = sum(1 for lg in runs
                        if (n := parse_expression(lg.best_expr)) is not None
                        and sympy_equivalent(n, NGUYEN_SYMPY[t]))
            rows.append((method, t, anytime, final, len(runs)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--budget", type=int, default=2_000_000)
    ap.add_argument("--targets", default="nguyen-3,nguyen-4")
    ap.add_argument("--max-lengths", default="30,64")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    max_lengths = [int(x) for x in args.max_lengths.split(",") if x.strip()]

    all_results = {}
    for L in max_lengths:
        common = dict(FAITHFUL, max_length=L)
        cfg = dict(
            budget=args.budget, seeds=args.seeds, n_points=20, x_range=[-1.0, 1.0],
            length_penalty=0.0, eps_success=1.0e-6, reward_mode="nrmse",
            log_every=50, track_symbolic=True, targets=targets,
            methods={
                "greedy": dict(proposer="greedy", **common),               # VPG
                "risk": dict(proposer="risk", mode="quantile", epsilon=0.05, **common),
            })
        log_dir = f"results/nguyen-collapse/maxlen{L}/logs"
        print(f"\n### max_length={L}  ({len(targets)} targets x 2 methods x "
              f"{args.seeds} seeds) ->  {log_dir}", flush=True)
        logs = run_experiment(cfg, log_dir=log_dir, resume=True, workers=args.workers)
        all_results[L] = recovery_table(logs, targets, args.seeds)

    print("\n\n================ Nguyen collapse sweep (SymPy recovery) ================")
    print("DSR paper (100 seeds): VPG N1-4 = 96/47/4/1% ; DSR = 100/100/100/100%")
    for L in max_lengths:
        print(f"\n  --- max_length = {L} ---")
        print(f"    {'method':7} {'target':9} {'anytime':>10} {'final-best':>12}")
        for method, t, anytime, final, n in all_results[L]:
            tag = "  (=VPG)" if method == "greedy" else ""
            print(f"    {method:7} {t:9} {anytime:>4}/{n:<4}   {final:>4}/{n:<4}{tag}")


if __name__ == "__main__":
    main()
