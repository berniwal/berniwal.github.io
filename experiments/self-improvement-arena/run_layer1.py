#!/usr/bin/env python3
"""Layer 1 entrypoint: LLM proposers on symbolic regression.

Three arms, all behind the SAME ask/tell loop, verifier, and tasks as Layer 0:
  - evolution    -- program-database in-context evolution (llm_evolution.py)
  - greedy_lora  -- greedy LoRA fine-tuning           (lora_proposer.py, arm=greedy)
  - risk_lora    -- risk-seeking LoRA fine-tuning     (lora_proposer.py, arm=risk)

Budget is in verifier calls = LLM generations (tens-to-hundreds), small compared
to Layer 0's 100k. LoRA gradient steps are free (no verifier budget).

SERIAL by design: MLX uses the unified-memory Metal GPU and one model saturates
it, so there is no --parallel. Crash-safe + resumable like Layer 0: each run's log
is checkpointed to disk and skipped on resume.

    python run_layer1.py --config configs/layer1_lora.yaml --out results_layer1_lora # sweep
    python run_layer1.py --arm greedy_lora --target medium --budget 64              # ad-hoc smoke
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from sia.expression import to_infix  # noqa: E402
from sia.metrics import RunLog, unique_fraction  # noqa: E402
from sia.plotting import make_layer1  # noqa: E402
from sia.task import make_task  # noqa: E402
from sia.verifier import Verifier  # noqa: E402
from layer1.llm_evolution import LLMEvolutionProposer, load_model  # noqa: E402
from layer1.lora_proposer import LoRAProposer  # noqa: E402

# Layer 1 arms. best_of_n is the no-learning floor (frozen weights, data-only
# prompt, no archive); evolution adds in-context learning; the LoRA arms add
# weight updates. So:  evolution - best_of_n = in-context lift; lora - best_of_n
# = fine-tuning lift.
ARMS = ("best_of_n", "evolution", "greedy_lora", "risk_lora")
# Arms that update model weights (LoRA mutates the model in place) -> each run
# needs its own freshly-loaded model; best_of_n and evolution reuse one (frozen).
LORA_ARMS = ("greedy_lora", "risk_lora")


def model_tag(model_id: str) -> str:
    """Short label, e.g. 'mlx-community/Qwen2.5-7B-Instruct-4bit' -> '7B'."""
    m = re.search(r"(\d+\.?\d*B)", model_id)
    return m.group(1) if m else model_id.split("/")[-1]


def build_proposer(arm, task, rng, model, tok, hp):
    """Construct the proposer for one of the arms from a single hp dict.
    Each proposer absorbs unused keys via **hp, so a merged dict is fine."""
    if arm == "best_of_n":  # control: evolution minus the archive (no learning)
        return LLMEvolutionProposer(task, rng, model, tok, use_archive=False, **hp)
    if arm == "evolution":
        return LLMEvolutionProposer(task, rng, model, tok, **hp)
    if arm == "greedy_lora":
        return LoRAProposer(task, rng, model, tok, arm="greedy", **hp)
    if arm == "risk_lora":
        return LoRAProposer(task, rng, model, tok, arm="risk", **hp)
    raise ValueError(f"unknown arm {arm!r}; choices: {list(ARMS)}")


def run_one(model, tok, arm, method, target, budget, seed, hp, verbose=True) -> RunLog:
    task = make_task(target, seed=seed)
    ver = Verifier(task)
    rng = np.random.default_rng(seed)
    proposer = build_proposer(arm, task, rng, model, tok, hp)
    log = RunLog(method=method, target=target, seed=seed, budget=budget)
    best, best_expr = 0.0, None
    while ver.calls < budget:
        cands = proposer.ask()
        results = [ver(c) for c in cands]
        proposer.tell(cands, results)
        for c, r in zip(cands, results):
            if r.reward > best:
                best, best_expr = r.reward, c
            if log.evals_to_solve is None and ver.success(c):
                log.success, log.evals_to_solve = True, ver.calls
        log.calls.append(ver.calls)
        log.best_reward.append(best)
        log.diversity.append(unique_fraction(cands))
        log.policy_entropy.append(proposer.diagnostics()["valid_fraction"])
        if verbose:
            print(f"    calls={ver.calls:4d}/{budget} best={best:.4f} "
                  f"top={to_infix(best_expr) if best_expr else '-'}", flush=True)
    log.best_expr = to_infix(best_expr) if best_expr is not None else ""
    return log


def arm_hp(config: dict, arm: str, temp: float) -> dict:
    """Merge shared generation hp with this arm's specific overrides + temperature.
    Per-arm overrides live under a top-level key named after the arm, e.g.
    ``greedy_lora: {lora_rank: 8, lr: 1.0e-4}``."""
    base = dict(batch_size=config["batch_size"], max_tokens=config["max_tokens"],
                n_data_shown=config.get("n_data_shown", 12))
    return dict(base, **config.get(arm, {}), temperature=temp)


def run_sweep(config: dict, out_dir: Path, resume: bool) -> None:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    arms = config.get("arms", ["evolution"])
    targets = config["targets"]
    temps = config["temperatures"]
    seeds = config["seeds"]
    budget = config["budget"]
    single_temp = len(temps) == 1

    total = len(config["models"]) * len(arms) * len(targets) * len(temps) * seeds
    done = 0
    logs: list[RunLog] = []
    t0 = time.time()

    # Models are the OUTER loop. The evolution arm reuses one loaded model (no
    # weight updates); the LoRA arms MUTATE the model when attaching the adapter,
    # so each LoRA run gets a freshly-loaded model (correct, at the cost of a
    # reload per run -- negligible vs. the tiny verifier-call budgets here).
    for model_id in config["models"]:
        tag = model_tag(model_id)
        shared = None        # lazily-loaded model reused by the evolution arm
        load_failed = False
        for arm in arms:
            for temp in temps:
                method = f"{tag}_{arm}" + ("" if single_temp else f"_t{temp}")
                for target in targets:
                    for seed in range(seeds):
                        done += 1
                        path = log_dir / f"{method}__{target}__seed{seed}.json"
                        if resume and path.exists():
                            try:
                                logs.append(RunLog(**json.loads(path.read_text())))
                                print(f"[{done}/{total}] skip {method}/{target}/seed{seed}",
                                      flush=True)
                                continue
                            except Exception:
                                pass
                        if load_failed:
                            continue
                        # acquire a model for this run
                        try:
                            if arm in LORA_ARMS:
                                model, tok = load_model(model_id)  # fresh: LoRA mutates
                            else:
                                if shared is None:
                                    print(f"Loading {model_id} ...", flush=True)
                                    tl = time.time()
                                    shared = load_model(model_id)
                                    print(f"  loaded in {time.time() - tl:.1f}s", flush=True)
                                model, tok = shared
                        except Exception as e:
                            print(f"[ERROR] could not load {model_id}: {e!r} -- skipping",
                                  flush=True)
                            load_failed = True
                            continue
                        el = time.time() - t0
                        print(f"[{done}/{total}] {method}/{target}/seed{seed} "
                              f"(elapsed {el/60:.0f}m)", flush=True)
                        try:
                            lg = run_one(model, tok, arm, method, target, budget, seed,
                                         arm_hp(config, arm, temp))
                        except Exception as e:
                            print(f"[ERROR] {method}/{target}/seed{seed}: {e!r}", flush=True)
                            continue
                        logs.append(lg)
                        tmp = path.with_suffix(".json.tmp")
                        tmp.write_text(json.dumps(lg.to_dict()))
                        tmp.replace(path)
                        if len(logs) % 5 == 0:  # refresh figures so partials are viewable
                            try:
                                make_layer1(logs, out_dir)
                            except Exception as e:
                                print(f"[warn] checkpoint plot failed: {e!r}", flush=True)

    if logs:
        print("\n" + make_layer1(logs, out_dir))
    print(f"Done in {(time.time() - t0)/60:.1f} min. Figures + table in {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML sweep config (else ad-hoc smoke)")
    ap.add_argument("--out", default=str(ROOT / "results_layer1"))
    ap.add_argument("--quick", action="store_true", help="tiny sweep to test the harness")
    ap.add_argument("--fresh", action="store_true", help="ignore existing logs, re-run all")
    # ad-hoc smoke args (used when --config is omitted)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--arm", default="evolution", choices=list(ARMS),
                    help="which Layer 1 arm to run in the ad-hoc smoke")
    ap.add_argument("--target", default="medium")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        if args.quick:
            config.update(config.get("quick", {}))
        out_dir = Path(args.out)
        arms = config.get("arms", ["evolution"])
        n = (len(config["models"]) * len(arms) * len(config["targets"])
             * len(config["temperatures"]) * config["seeds"])
        print(f"Layer 1 sweep: {n} runs "
              f"({len(config['models'])} models x {len(arms)} arms "
              f"x {len(config['targets'])} targets "
              f"x {len(config['temperatures'])} temps x {config['seeds']} seeds), "
              f"budget={config['budget']} generations each. SERIAL (MLX).", flush=True)
        run_sweep(config, out_dir, resume=not args.fresh)
        return

    # ad-hoc single-model, single-arm smoke. LoRA arms mutate the model, so reload
    # per seed for a clean adapter; evolution can reuse one load.
    hp = dict(batch_size=args.batch_size, temperature=args.temperature,
              max_tokens=args.max_tokens)
    method = f"{model_tag(args.model)}_{args.arm}"
    shared = None
    for seed in range(args.seeds):
        print(f"=== {args.arm} on {args.target}, seed {seed} (budget {args.budget}) ===",
              flush=True)
        if args.arm in LORA_ARMS:
            model, tok = load_model(args.model)
        else:
            if shared is None:
                print(f"Loading {args.model} ...", flush=True)
                t0 = time.time()
                shared = load_model(args.model)
                print(f"Loaded in {time.time() - t0:.1f}s\n", flush=True)
            model, tok = shared
        log = run_one(model, tok, args.arm, method, args.target, args.budget, seed, hp)
        print(f"--> success={log.success} best_reward={log.best_reward[-1]:.4f} "
              f"best={log.best_expr}\n", flush=True)


if __name__ == "__main__":
    main()
