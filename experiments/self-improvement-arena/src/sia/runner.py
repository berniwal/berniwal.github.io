"""The runner: drive a proposer's ask/tell loop against the shared verifier under
a fixed verifier-call budget, logging best-so-far reward and diversity. Same loop
for every method -> the only difference between methods is the proposer.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .expression import to_infix
from .metrics import RunLog, unique_fraction
from .proposers import get_proposer
from .task import make_task
from .verifier import Verifier


def run_method(method: str, proposer_name: str, target: str, budget: int, seed: int,
               proposer_hp: dict, n_points: int = 30, x_range=(-3.0, 3.0),
               length_penalty: float = 0.001, eps_success: float = 1e-6,
               log_every: int = 1) -> RunLog:
    # Same dataset for every method at a given (target, seed) -> a fair contest.
    task = make_task(target, n_points=n_points, x_range=tuple(x_range), seed=seed)
    ver = Verifier(task, length_penalty=length_penalty, eps_success=eps_success)
    rng = np.random.default_rng(seed)
    proposer = get_proposer(proposer_name)(task, rng, **proposer_hp)

    log = RunLog(method=method, target=target, seed=seed, budget=budget)
    best = 0.0
    best_expr = None
    has_diag = hasattr(proposer, "diagnostics")
    batch_idx = 0

    while ver.calls < budget:
        candidates = proposer.ask()
        results = [ver(c) for c in candidates]
        proposer.tell(candidates, results)
        batch_idx += 1

        for c, r in zip(candidates, results):
            if r.reward > best:
                best, best_expr = r.reward, c
            # success/evals-to-solve is exact regardless of log_every (checked here)
            if log.evals_to_solve is None and ver.success(c):
                log.success = True
                log.evals_to_solve = ver.calls

        # Downsample the curve for very long runs; the final point is always kept.
        if batch_idx % log_every == 0 or ver.calls >= budget:
            log.calls.append(ver.calls)
            log.best_reward.append(best)
            log.diversity.append(unique_fraction(candidates))
            log.policy_entropy.append(
                proposer.diagnostics()["policy_entropy"] if has_diag else float("nan"))

    log.best_expr = to_infix(best_expr) if best_expr is not None else ""
    return log


def run_experiment(config: dict, log_dir=None, resume: bool = True,
                   verbose: bool = True, checkpoint_cb=None,
                   checkpoint_every: int = 25) -> list[RunLog]:
    """Run every (method, target, seed). Crash-safe and resumable:

    - each run's log is written to ``log_dir`` *immediately* on completion;
    - on resume, a run whose log file already exists is loaded and skipped;
    - a failing run is logged to stderr and skipped (so it is retried next resume),
      never killing the batch.

    Rerunning the same command continues from where a crash left off.
    """
    logs: list[RunLog] = []
    total = len(config["targets"]) * len(config["methods"]) * config["seeds"]
    log_dir = Path(log_dir) if log_dir is not None else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    for target in config["targets"]:
        for method, mhp in config["methods"].items():
            hp = dict(mhp)
            proposer_name = hp.pop("proposer")
            for seed in range(config["seeds"]):
                done += 1
                path = (log_dir / f"{method}__{target}__seed{seed}.json"
                        if log_dir is not None else None)
                if resume and path is not None and path.exists():
                    try:
                        logs.append(RunLog(**json.loads(path.read_text())))
                        if verbose:
                            _progress(done, total, t0, f"skip {method}/{target}/seed{seed}")
                        continue
                    except Exception:  # corrupt/partial file -> re-run it
                        pass
                try:
                    lg = run_method(
                        method=method, proposer_name=proposer_name, target=target,
                        budget=config["budget"], seed=seed, proposer_hp=hp,
                        n_points=config["n_points"], x_range=config["x_range"],
                        length_penalty=config["length_penalty"],
                        eps_success=float(config["eps_success"]),
                        log_every=config.get("log_every", 1),
                    )
                except Exception as e:  # one bad run must not kill the batch
                    sys.stderr.write(f"\n[ERROR] {method}/{target}/seed{seed}: {e!r}\n")
                    continue
                logs.append(lg)
                if path is not None:  # atomic write so a crash mid-write can't corrupt
                    tmp = path.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(lg.to_dict()))
                    tmp.replace(path)
                if verbose:
                    _progress(done, total, t0, f"{method}/{target}/seed{seed}")
                if checkpoint_cb is not None and done % checkpoint_every == 0:
                    try:
                        checkpoint_cb(logs)  # regenerate figures so partial results are viewable
                    except Exception as e:
                        sys.stderr.write(f"\n[warn] checkpoint plot failed: {e!r}\n")
    if verbose:
        sys.stderr.write("\n")
    return logs


def _progress(done: int, total: int, t0: float, label: str) -> None:
    """Dependency-free progress line (no tqdm): bar + count + elapsed + ETA."""
    frac = done / total
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done)
    bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
    sys.stderr.write(
        f"\r[{bar}] {done}/{total}  elapsed {elapsed:5.0f}s  eta {eta:5.0f}s  {label:28s}")
    sys.stderr.flush()
