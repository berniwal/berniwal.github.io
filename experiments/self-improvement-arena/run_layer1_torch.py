#!/usr/bin/env python3
"""Drive the PyTorch GRPO LLM proposer (Layer 1) on a symbolic-regression target.

Runs N rounds of: sample a batch from the LLM -> score with the shared Verifier ->
GRPO update (per arm, via shared objectives). Logs best-so-far reward, valid fraction,
and recovery (numeric MSE + exact SymPy) per round; writes JSON + prints a summary.
Built to run on a GPU pod (see experiments/runpod/run_layer1_torch.sh).

    python3 run_layer1_torch.py --target medium --arm risk --rounds 40 --batch 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC)
from sia.expression import sympy_equivalent, to_infix  # noqa: E402
from sia.task import make_task  # noqa: E402
from sia.verifier import Verifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--target", default="medium")
    ap.add_argument("--arm", default="risk", choices=["greedy", "risk"])
    ap.add_argument("--mode", default="quantile", choices=["quantile", "entropic", "cvar"])
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--clip-low", type=float, default=0.2)
    ap.add_argument("--clip-high", type=float, default=0.28)
    ap.add_argument("--trunc-is", type=float, default=2.0)
    ap.add_argument("--std-normalize", action="store_true")
    ap.add_argument("--no-const", action="store_true", help="disable const placeholder + BFGS")
    ap.add_argument("--reward-mode", default="nrmse", choices=["mse", "nrmse"])
    ap.add_argument("--n-points", type=int, default=20)
    ap.add_argument("--x-range", default="-1,1")
    ap.add_argument("--eps-success", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/layer1-torch")
    args = ap.parse_args()

    lo, hi = (float(v) for v in args.x_range.split(","))
    task = make_task(args.target, n_points=args.n_points, x_range=(lo, hi), seed=args.seed)
    ver = Verifier(task, eps_success=args.eps_success, reward_mode=args.reward_mode)

    from layer1.torch_lora_proposer import TorchLoRAProposer
    prop = TorchLoRAProposer(
        task, model_id=args.model, arm=args.arm, mode=args.mode, epsilon=args.epsilon,
        beta=args.beta, batch_size=args.batch, lr=args.lr, weight_decay=args.weight_decay,
        lora_rank=args.lora_rank, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, const_placeholder=not args.no_const,
        ppo_epochs=args.ppo_epochs, clip_low=args.clip_low, clip_high=args.clip_high,
        trunc_is=args.trunc_is, std_normalize=args.std_normalize, seed=args.seed)

    best, best_expr = 0.0, ""
    num_solved_at = sym_solved_at = None
    history = []
    t0 = time.time()
    for rd in range(1, args.rounds + 1):
        cands = prop.ask()
        results = [ver(c) for c in cands]
        for c, r in zip(cands, results):
            if r.reward > best:
                best, best_expr = r.reward, to_infix(c)
            if num_solved_at is None and ver.success(c):
                num_solved_at = ver.calls
            if sym_solved_at is None and task.target_sympy and ver.success(c) \
                    and sympy_equivalent(c, task.target_sympy):
                sym_solved_at = ver.calls
        prop.tell(cands, results)
        d = prop.diagnostics()
        history.append(dict(round=rd, calls=ver.calls, best=best,
                            valid=d["valid_fraction"], loss=d["loss"],
                            batch_best=max(r.reward for r in results)))
        print(f"[r{rd:03d}] calls={ver.calls} best={best:.4f} "
              f"batch_best={max(r.reward for r in results):.4f} "
              f"valid={d['valid_fraction']:.2f} loss={d['loss']:.4f} "
              f"num_solved={num_solved_at} sym_solved={sym_solved_at}", flush=True)

    out_dir = os.path.join(args.out, f"{args.arm}-{args.mode}-{args.target}-seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    summary = dict(args=vars(args), best=best, best_expr=best_expr,
                   numeric_solved_at=num_solved_at, symbolic_solved_at=sym_solved_at,
                   elapsed_s=round(time.time() - t0, 1), history=history)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n=== {args.arm}/{args.mode} on {args.target} ===")
    print(f"best={best:.4f} expr={best_expr}")
    print(f"numeric_solved_at={num_solved_at} symbolic_solved_at={sym_solved_at} "
          f"({summary['elapsed_s']}s) -> {out_dir}")


if __name__ == "__main__":
    main()
