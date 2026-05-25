"""The runner: drive a proposer's ask/tell loop against the shared verifier under
a fixed verifier-call budget, logging best-so-far reward and diversity. Same loop
for every method -> the only difference between methods is the proposer.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .expression import sympy_equivalent, to_infix
from .metrics import RunLog, unique_fraction
from .proposers import get_proposer
from .task import make_task
from .verifier import Verifier


def available_cpus() -> int:
    """Best-effort count of *usable* CPUs in this (possibly containerized) env.

    ``os.cpu_count()`` reports the host's cores, which over-counts inside a quota- or
    affinity-limited container (e.g. a "28 vCPU" RunPod pod on a 64-core host). Take
    the min of: the CPU-affinity mask, the cgroup CPU quota (v2 then v1), and
    ``os.cpu_count()`` -- so a parallel run neither oversubscribes a capped container
    nor overcounts when pinned to a subset of cores.
    """
    counts: list[int] = []
    n = os.cpu_count()
    if n:
        counts.append(n)
    try:
        counts.append(len(os.sched_getaffinity(0)))  # Linux; respects core pinning
    except AttributeError:
        pass  # not on macOS/Windows
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            counts.append(max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and p > 0:
            counts.append(max(1, q // p))
    except (OSError, ValueError):
        pass
    return min(counts) if counts else 1


def run_method(method: str, proposer_name: str, target: str, budget: int, seed: int,
               proposer_hp: dict, n_points: int = 30, x_range=(-3.0, 3.0),
               length_penalty: float = 0.001, eps_success: float = 1e-6,
               reward_mode: str = "mse", log_every: int = 1,
               track_symbolic: bool = False) -> RunLog:
    # Same dataset for every method at a given (target, seed) -> a fair contest.
    task = make_task(target, n_points=n_points, x_range=tuple(x_range), seed=seed)
    ver = Verifier(task, length_penalty=length_penalty, eps_success=eps_success,
                   reward_mode=reward_mode)
    rng = np.random.default_rng(seed)
    proposer = get_proposer(proposer_name)(task, rng, **proposer_hp)

    log = RunLog(method=method, target=target, seed=seed, budget=budget)
    best = 0.0
    best_expr = None
    has_diag = hasattr(proposer, "diagnostics")
    batch_idx = 0
    # Cache CAS verdicts per distinct expression so the (rare, gated) symbolic
    # check never re-simplifies the same candidate twice within a run.
    sym_cache: dict[str, bool] = {}

    while ver.calls < budget:
        candidates = proposer.ask()
        results = [ver(c) for c in candidates]
        proposer.tell(candidates, results)
        batch_idx += 1

        for c, r in zip(candidates, results):
            if r.reward > best:
                best, best_expr = r.reward, c
            # NUMERIC recovery: held-out MSE < eps_success. Exact regardless of
            # log_every (checked here). ver.success does not consume budget.
            passed = ver.success(c)
            if log.evals_to_solve is None and passed:
                log.success = True
                log.evals_to_solve = ver.calls
            # SYMBOLIC recovery (DSR's strict criterion): gate on the cheap numeric
            # pass first (an exactly-equivalent expr always fits to ~0), then ask the
            # CAS. Cache per infix so repeated near-fits don't re-trigger SymPy.
            if track_symbolic and log.evals_to_solve_symbolic is None and passed:
                key = to_infix(c)
                eq = sym_cache.get(key)
                if eq is None:
                    eq = sympy_equivalent(c, task.target_sympy)
                    sym_cache[key] = eq
                if eq:
                    log.success_symbolic = True
                    log.evals_to_solve_symbolic = ver.calls

        # Downsample the curve for very long runs; the final point is always kept.
        if batch_idx % log_every == 0 or ver.calls >= budget:
            log.calls.append(ver.calls)
            log.best_reward.append(best)
            log.diversity.append(unique_fraction(candidates))
            log.policy_entropy.append(
                proposer.diagnostics()["policy_entropy"] if has_diag else float("nan"))

    log.best_expr = to_infix(best_expr) if best_expr is not None else ""
    return log


def _run_job(job: dict) -> RunLog:
    """Run one (method, target, seed) and atomically checkpoint its log to disk.
    Top-level so it is picklable for the process pool."""
    lg = run_method(
        method=job["method"], proposer_name=job["proposer_name"],
        target=job["target"], budget=job["budget"], seed=job["seed"],
        proposer_hp=job["hp"], n_points=job["n_points"], x_range=job["x_range"],
        length_penalty=job["length_penalty"], eps_success=job["eps_success"],
        reward_mode=job["reward_mode"], log_every=job["log_every"],
        track_symbolic=job.get("track_symbolic", False),
    )
    if job["path"] is not None:  # atomic write -> a crash mid-write can't corrupt
        path = Path(job["path"])
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(lg.to_dict()))
        tmp.replace(path)
    return lg


def run_experiment(config: dict, log_dir=None, resume: bool = True,
                   verbose: bool = True, checkpoint_cb=None,
                   checkpoint_every: int = 25, workers: int = 1) -> list[RunLog]:
    """Run every (method, target, seed). Crash-safe, resumable, optionally parallel:

    - each run's log is written to ``log_dir`` *immediately* on completion;
    - on resume, a run whose log file already exists is loaded and skipped;
    - a failing run is logged to stderr and skipped (so it is retried next resume),
      never killing the batch;
    - ``workers > 1`` runs the independent jobs across processes (results are
      identical -- each run is seeded by its own seed, independent of scheduling).

    Rerunning the same command continues from where a crash left off.
    """
    logs: list[RunLog] = []
    total = len(config["targets"]) * len(config["methods"]) * config["seeds"]
    log_dir = Path(log_dir) if log_dir is not None else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Build the job list, loading (and skipping) any already-completed runs.
    jobs: list[dict] = []
    for target in config["targets"]:
        for method, mhp in config["methods"].items():
            hp = dict(mhp)
            proposer_name = hp.pop("proposer")
            for seed in range(config["seeds"]):
                path = (log_dir / f"{method}__{target}__seed{seed}.json"
                        if log_dir is not None else None)
                if resume and path is not None and path.exists():
                    try:
                        logs.append(RunLog(**json.loads(path.read_text())))
                        continue
                    except Exception:  # corrupt/partial file -> re-run it
                        pass
                jobs.append(dict(
                    method=method, proposer_name=proposer_name, target=target,
                    seed=seed, hp=hp, path=(str(path) if path else None),
                    budget=config["budget"], n_points=config["n_points"],
                    x_range=config["x_range"], length_penalty=config["length_penalty"],
                    eps_success=float(config["eps_success"]),
                    reward_mode=config.get("reward_mode", "mse"),
                    log_every=config.get("log_every", 1),
                    track_symbolic=config.get("track_symbolic", False),
                ))

    t0 = time.time()
    done = len(logs)
    if verbose and done:
        _progress(done, total, t0, f"resumed {done} runs")

    def _accept(lg: RunLog) -> None:
        nonlocal done
        done += 1
        logs.append(lg)
        if verbose:
            _progress(done, total, t0, f"{lg.method}/{lg.target}/seed{lg.seed}")
        if checkpoint_cb is not None and done % checkpoint_every == 0:
            try:
                checkpoint_cb(logs)  # regenerate figures so partial results are viewable
            except Exception as e:
                sys.stderr.write(f"\n[warn] checkpoint plot failed: {e!r}\n")

    if workers and workers > 1 and jobs:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_job, j): j for j in jobs}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    _accept(fut.result())
                except Exception as e:  # one bad run must not kill the batch
                    sys.stderr.write(
                        f"\n[ERROR] {j['method']}/{j['target']}/seed{j['seed']}: {e!r}\n")
    else:
        for j in jobs:
            try:
                _accept(_run_job(j))
            except Exception as e:
                sys.stderr.write(
                    f"\n[ERROR] {j['method']}/{j['target']}/seed{j['seed']}: {e!r}\n")

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
