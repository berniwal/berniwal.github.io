"""Analyse the PUCT 2x2 ablation: with/without GRPO training, with/without
PUCT search.

Pulls summary.json for each of the 9 (+1 existing) seeds from GCS, plus the
existing `risk + reasoning` and `best_of_n + reasoning` data on disk, and
emits a publishable table that includes:

  - numeric-solved fraction (n/5)
  - DSR-symbolic-recovered fraction (n/5)
  - strict-symbolic-recovered fraction (n/5)
  - mean best (over 5 seeds)
  - mean num_solved_at calls (for seeds that did solve)
  - "found earlier with GRPO" delta on shared-solve seeds

Pass --update-blog-data to also write a JSON file for the blog widget at
public/data/symbolic-regression/puct_ablation.json.

Usage:
  python3 analyze_puct_ablation.py                  # print table to stdout
  python3 analyze_puct_ablation.py --update-blog-data  # also dump JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, "src")
import sympy as sp
from sia.expression import parse_expression, sympy_equivalent, to_sympy
from sia.task import make_task

DEFAULT_TARGET = "harder"

# (cell, seed) -> GCS exp name
TRAINED_EXP = {s: f"l1-q3-puct-seed{s}-2026-05-28" for s in range(5)}
TRAINED_EXP[0] = "l1-q3-puct-full2-seed0-2026-05-28"  # the existing seed-0 run
NOTRAIN_EXP = {s: f"l1-q3-puct-notrain-seed{s}-2026-05-28" for s in range(5)}


@dataclass
class SeedSummary:
    seed: int
    best: float
    best_expr: str
    num_solved_at: int | None
    dsr_symbolic: bool
    strict_symbolic: bool
    rounds_done: int


def _gcs():
    from google.cloud import storage  # type: ignore
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                          os.path.expanduser("~/runpod/sa-runpod-experiments.json"))
    return storage.Client().bucket("runpod-experiments")


def _strict_ok(node, target_sympy):
    """sympy_equivalent AND raw_terms == target_terms (no junk x^2 padding)."""
    if node is None:
        return False
    try:
        raw = to_sympy(node)
        n_raw = len(sp.expand(raw).args) if isinstance(sp.expand(raw), sp.Add) else 1
        n_tgt = len(sp.expand(target_sympy).args) if isinstance(sp.expand(target_sympy), sp.Add) else 1
        return sympy_equivalent(node, target_sympy, const_tol=1e-3) and n_raw == n_tgt
    except Exception:
        return False


def _summarise_one(bucket, exp: str, arm: str, target_sympy) -> SeedSummary | None:
    path = f"{exp}/results/layer1-torch/{arm}-quantile-{DEFAULT_TARGET}-seed{exp[-1]}/summary.json"
    # The seed in the path is whatever the bash wrapper used; for our exp naming
    # the seed digit appears at the end of the seed-specific exp prefix.
    # Find the actual summary path via list.
    seed_digit = int(exp.split("seed")[-1].split("-")[0])
    path = f"{exp}/results/layer1-torch/{arm}-quantile-{DEFAULT_TARGET}-seed{seed_digit}/summary.json"
    blob = bucket.blob(path)
    if not blob.exists():
        return None
    d = json.loads(blob.download_as_text())
    node = parse_expression(d.get("best_expr", ""))
    dsr_ok = node is not None and sympy_equivalent(node, target_sympy, const_tol=1e-3)
    strict_ok = _strict_ok(node, target_sympy)
    return SeedSummary(
        seed=seed_digit,
        best=d.get("best", 0.0),
        best_expr=d.get("best_expr", ""),
        num_solved_at=d.get("numeric_solved_at"),
        dsr_symbolic=dsr_ok,
        strict_symbolic=strict_ok,
        rounds_done=len(d.get("history", [])),
    )


def _cell_metrics(seed_summaries: list[SeedSummary]) -> dict:
    if not seed_summaries:
        return {"n": 0, "complete": 0}
    n = len(seed_summaries)
    num = sum(1 for s in seed_summaries if isinstance(s.num_solved_at, int))
    dsr = sum(1 for s in seed_summaries if s.dsr_symbolic)
    strict = sum(1 for s in seed_summaries if s.strict_symbolic)
    mean_best = sum(s.best for s in seed_summaries) / n
    solve_calls = [s.num_solved_at for s in seed_summaries
                   if isinstance(s.num_solved_at, int)]
    mean_solve_calls = sum(solve_calls) / len(solve_calls) if solve_calls else None
    return {
        "n": n,
        "complete": sum(1 for s in seed_summaries if s.rounds_done >= 80),
        "numeric": num,
        "dsr_symbolic": dsr,
        "strict_symbolic": strict,
        "mean_best": round(mean_best, 4),
        "mean_solve_calls": (round(mean_solve_calls, 1)
                              if mean_solve_calls is not None else None),
        "per_seed": [{
            "seed": s.seed, "best": round(s.best, 4),
            "num_solved_at": s.num_solved_at,
            "dsr": s.dsr_symbolic, "strict": s.strict_symbolic,
            "best_expr": s.best_expr,
        } for s in seed_summaries],
    }


def _gather(bucket, target_sympy):
    cells = {}
    trained = []
    for s, exp in TRAINED_EXP.items():
        r = _summarise_one(bucket, exp, "puct", target_sympy)
        if r is not None:
            trained.append(r)
    notrain = []
    for s, exp in NOTRAIN_EXP.items():
        r = _summarise_one(bucket, exp, "puct", target_sympy)
        if r is not None:
            notrain.append(r)
    cells["puct + GRPO"] = _cell_metrics(trained)
    cells["puct + lr=0 (untrained)"] = _cell_metrics(notrain)
    return cells


def _per_seed_compare(trained: list[SeedSummary], notrain: list[SeedSummary]):
    """For seeds with both variants: did GRPO find the solution earlier?"""
    t = {s.seed: s for s in trained}
    n = {s.seed: s for s in notrain}
    shared = sorted(set(t) & set(n))
    rows = []
    for seed in shared:
        ts, ns = t[seed], n[seed]
        rows.append(dict(
            seed=seed,
            trained_num_solved_at=ts.num_solved_at,
            notrain_num_solved_at=ns.num_solved_at,
            trained_best=ts.best,
            notrain_best=ns.best,
            trained_strict=ts.strict_symbolic,
            notrain_strict=ns.strict_symbolic,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-blog-data", action="store_true")
    args = ap.parse_args()

    bucket = _gcs()
    task = make_task(DEFAULT_TARGET, n_points=20, x_range=(-1, 1), seed=0)
    target_sympy = task.target_sympy

    cells = _gather(bucket, target_sympy)
    print(f"\n=== PUCT 2x2 ablation on `{DEFAULT_TARGET}` ===\n")
    fmt = "{:<28} | {:>10} | {:>10} | {:>14} | {:>10} | {:>14}"
    print(fmt.format("cell", "numeric", "DSR-sym", "STRICT-sym",
                     "mean best", "mean solve calls"))
    print("-" * 100)
    for name, c in cells.items():
        if c["n"] == 0:
            print(fmt.format(name, "—", "—", "—", "—", "—"))
            continue
        num = f"{c['numeric']}/{c['n']}"
        dsr = f"{c['dsr_symbolic']}/{c['n']}"
        st = f"{c['strict_symbolic']}/{c['n']}"
        mb = f"{c['mean_best']:.4f}"
        sc = (f"{c['mean_solve_calls']:.0f}"
              if c['mean_solve_calls'] is not None else "—")
        print(fmt.format(name, num, dsr, st, mb, sc))
    print()

    # Per-seed comparison: who found it earlier?
    trained = []
    for exp in TRAINED_EXP.values():
        r = _summarise_one(bucket, exp, "puct", target_sympy)
        if r is not None:
            trained.append(r)
    notrain = []
    for exp in NOTRAIN_EXP.values():
        r = _summarise_one(bucket, exp, "puct", target_sympy)
        if r is not None:
            notrain.append(r)

    rows = _per_seed_compare(trained, notrain)
    if rows:
        print("=== per-seed comparison (seeds where both ran) ===")
        cf = "{:>4} | {:>14} | {:>14} | {:>9} | {:>9} | {:>16}"
        print(cf.format("seed", "trained solve@",
                        "untrained solve@", "tr best", "un best", "GRPO earlier by"))
        print("-" * 80)
        for r in rows:
            t_calls = r["trained_num_solved_at"]
            n_calls = r["notrain_num_solved_at"]
            delta = "—"
            if isinstance(t_calls, int) and isinstance(n_calls, int):
                d = n_calls - t_calls
                if d > 0:
                    delta = f"+{d} calls"
                elif d < 0:
                    delta = f"{d} calls (untrained faster)"
                else:
                    delta = "tied"
            elif isinstance(t_calls, int) and n_calls is None:
                delta = "trained-only solve"
            elif isinstance(n_calls, int) and t_calls is None:
                delta = "untrained-only solve"
            print(cf.format(
                r["seed"],
                str(t_calls) if t_calls is not None else "—",
                str(n_calls) if n_calls is not None else "—",
                f"{r['trained_best']:.4f}",
                f"{r['notrain_best']:.4f}",
                delta,
            ))

    if args.update_blog_data:
        blog_path = "../../public/data/symbolic-regression/puct_ablation.json"
        blog_path = os.path.abspath(blog_path)
        os.makedirs(os.path.dirname(blog_path), exist_ok=True)
        with open(blog_path, "w") as f:
            json.dump({"target": DEFAULT_TARGET, "cells": cells,
                       "per_seed_compare": rows}, f, indent=2)
        print(f"\nwrote blog data -> {blog_path}")


if __name__ == "__main__":
    main()
