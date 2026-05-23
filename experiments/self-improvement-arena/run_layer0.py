#!/usr/bin/env python3
"""Single entrypoint that reproduces every Layer 0 figure + the results table.

    python run_layer0.py --config configs/layer0.yaml   # full run
    python run_layer0.py --quick                         # fast smoke run

Writes results/{curves,diversity,success}.png, results/table.md, and the raw
per-run logs to results/logs/*.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sia.plotting import make_all  # noqa: E402
from sia.runner import run_experiment  # noqa: E402

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "layer0.yaml"))
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget + few seeds for a fast smoke test")
    ap.add_argument("--out", default=None,
                    help="output dir (default: results/, or results_quick/ with --quick)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing logs and re-run everything (default: resume)")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.quick:
        config.update(config.get("quick", {}))
    # --quick writes to a separate dir so a smoke run never clobbers the
    # committed headline figures/table.
    if args.out is None:
        args.out = str(ROOT / ("results_quick" if args.quick else "results"))

    out_dir = Path(args.out)
    log_dir = out_dir / "logs"

    n_runs = len(config["targets"]) * len(config["methods"]) * config["seeds"]
    done_already = len(list(log_dir.glob("*.json"))) if log_dir.exists() else 0
    print(f"Running {n_runs} runs "
          f"({len(config['methods'])} methods x {len(config['targets'])} targets "
          f"x {config['seeds']} seeds), budget={config['budget']} verifier calls each.")
    if not args.fresh and done_already:
        print(f"Resuming: {done_already}/{n_runs} runs already on disk in {log_dir}/ "
              f"will be skipped.")

    t0 = time.time()
    # Regenerate figures/table periodically so partial results are viewable even
    # mid-run; logs are saved per-run so a crash is safe to resume (rerun the
    # same command).
    logs = run_experiment(config, log_dir=log_dir, resume=not args.fresh,
                          checkpoint_cb=lambda lg: make_all(lg, out_dir))
    print(f"Done in {time.time() - t0:.1f}s.")

    table = make_all(logs, out_dir)
    print(f"\nWrote figures + table to {out_dir}/\n")
    print(table)


if __name__ == "__main__":
    main()
