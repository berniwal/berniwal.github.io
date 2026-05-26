#!/usr/bin/env python3
"""Bundle the arena's pre-computed results into the blog's static-asset tree, so the
React post (src/posts/VisualizingSymbolicRegression.jsx) can render the interactive
"watch them evolve" widget without running any Python.

Two data sources, normalized to ONE shared shape so Layer 0 (numpy proposers) and
Layer 1 (LLM proposer) appear in the SAME format in the widget:

  Layer 0  -- results/layer0_<reward>/replay.json   (made by export_replay.py;
              one representative seed, per-batch checkpoints with the fit overlay).
  Layer 1  -- results/layer1-torch/<arm>-<mode>-<target>-seed<seed>/summary.json
              (synced from RunPod via experiments/runpod/fetch.py). Per-round history
              is normalized into the same checkpoint shape; best_expr is parsed +
              evaluated on the target's x-grid to get the fit overlay, exactly like
              Layer 0. Recovery (numeric/symbolic solved) is aggregated across seeds.

Writes:
  public/data/symbolic-regression/layer0_mse.json     (copied verbatim)
  public/data/symbolic-regression/layer0_nrmse.json   (copied verbatim)
  public/data/symbolic-regression/layer1.json         (only if summaries are present;
                                                        otherwise left untouched)

    python export_blog_data.py                 # bundle whatever results exist
    python export_blog_data.py --layer1-only   # re-derive only layer1.json
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from sia.expression import evaluate, parse_expression  # noqa: E402

BLOG_DATA = ROOT.parent.parent / "public" / "data" / "symbolic-regression"

# x-grids must match export_replay.py so the two layers overlay on the same axis.
N_GRID = 60
X_RANGE = {"easy": (-3.0, 3.0), "medium": (-3.0, 3.0), "harder": (-3.0, 3.0)}

# Pretty labels + colours for the Layer-1 arms (kept parallel to the Layer-0 STYLE).
LAYER1_STYLE = {
    "greedy":    ("#d1495b", "Greedy LoRA (E[R])"),
    "risk":      ("#2a9d8f", "Risk-seeking LoRA (DSR)"),
    "entropic":  ("#8d6cab", "Entropic LoRA (Jβ)"),  # risk + mode=entropic collapses to "entropic"
    "best_of_n": ("#e9c46a", "Best-of-N (no training)"),
    "evolution": ("#577590", "Evolution (in-context DB)"),
}
# entropic shares the "risk" family but a distinct colour
LAYER1_MODE_LABEL = {"quantile": "risk-seeking", "entropic": "entropic J_beta",
                     "cvar": "risk-averse"}


def _round(values, n: int = 4):
    out = []
    for v in values:
        v = float(v)
        out.append(round(v, n) if math.isfinite(v) else None)
    return out


def _scalar(v, n: int = 4):
    if v is None:
        return None
    v = float(v)
    return round(v, n) if math.isfinite(v) else None


def copy_layer0() -> list[str]:
    """Copy the two Layer-0 replays into the blog tree verbatim."""
    copied = []
    for reward in ("mse", "nrmse"):
        src = ROOT / "results" / f"layer0_{reward}" / "replay.json"
        if not src.exists():
            print(f"  SKIP layer0_{reward}: {src} missing "
                  f"(run `python export_replay.py` first)")
            continue
        dst = BLOG_DATA / f"layer0_{reward}.json"
        shutil.copyfile(src, dst)
        copied.append(dst.name)
        print(f"  copied {src.name} -> {dst.relative_to(ROOT.parent.parent)} "
              f"({dst.stat().st_size / 1024:.0f} KB)")
    return copied


def _arm_key(arm: str, mode: str) -> str:
    """Collapse (arm, mode) into one widget series id."""
    if arm == "risk" and mode == "entropic":
        return "entropic"
    return arm


def _curve_from_history(history: list[dict], x_grid: np.ndarray) -> list[dict]:
    """Normalize a torch run's per-round history into the shared checkpoint shape."""
    out = []
    for h in history:
        expr = h.get("best_expr") or ""
        y_pred = None
        if expr:
            node = parse_expression(expr, const_placeholder=False)
            if node is not None:
                with np.errstate(all="ignore"):
                    y_pred = _round(evaluate(node, x_grid))
        out.append({
            "calls": int(h.get("calls", 0)),
            "best": _scalar(h.get("best")),
            "batch_mean": _scalar(h.get("batch_mean")),
            "best_infix": expr,
            "y_pred": y_pred,
        })
    return out


def build_layer1() -> dict | None:
    """Aggregate results/layer1-torch/*/summary.json into the widget schema.

    Returns None if no summaries are present (caller leaves layer1.json untouched).
    """
    src_dir = ROOT / "results" / "layer1-torch"
    summaries = sorted(src_dir.glob("*/summary.json")) if src_dir.exists() else []
    if not summaries:
        print("  no Layer-1 summaries under results/layer1-torch/ "
              "(fetch from GCS with experiments/runpod/fetch.py) -- skipping layer1.json")
        return None

    # group runs by (target, arm_key); collect per-seed best + recovery + one curve
    groups: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"seeds": [], "curve": None, "model": None})
    targets_seen: dict[str, dict] = {}

    for path in summaries:
        s = json.loads(path.read_text())
        a = s.get("args", {})
        target = a.get("target", "?")
        arm = a.get("arm", "?")
        mode = a.get("mode", "quantile")
        key = (target, _arm_key(arm, mode))
        g = groups[key]
        g["model"] = a.get("model", g["model"])
        g["seeds"].append({
            "seed": a.get("seed"),
            "best": _scalar(s.get("best")),
            "numeric_solved_at": s.get("numeric_solved_at"),
            "symbolic_solved_at": s.get("symbolic_solved_at"),
        })
        # keep the best-performing seed's curve as the representative trace
        if g["curve"] is None or (_scalar(s.get("best")) or 0) > g["curve"]["best"]:
            lo, hi = X_RANGE.get(target, (-3.0, 3.0))
            xs = np.linspace(lo, hi, N_GRID)
            g["curve"] = {"best": _scalar(s.get("best")) or 0,
                          "checkpoints": _curve_from_history(s.get("history", []), xs)}

        # record the target's x-grid / target curve once
        if target not in targets_seen:
            lo, hi = X_RANGE.get(target, (-3.0, 3.0))
            xs = np.linspace(lo, hi, N_GRID)
            tgt_expr = a.get("target")  # target name, not infix; fill infix below if known
            targets_seen[target] = {"x_grid": _round(xs), "target_name": tgt_expr}

    out_targets: dict[str, dict] = {}
    for (target, akey), g in groups.items():
        n = len(g["seeds"])
        num = sum(1 for s in g["seeds"] if s["numeric_solved_at"] is not None)
        sym = sum(1 for s in g["seeds"] if s["symbolic_solved_at"] is not None)
        color, label = LAYER1_STYLE.get(akey, ("#888", akey))
        entry = out_targets.setdefault(target, {**targets_seen[target], "arms": {}})
        entry["arms"][akey] = {
            "label": label, "color": color,
            "seeds": n, "numeric_solved": num, "symbolic_solved": sym,
            "mean_best": _scalar(np.mean([s["best"] for s in g["seeds"]
                                          if s["best"] is not None]) if n else None),
            "checkpoints": g["curve"]["checkpoints"] if g["curve"] else [],
        }

    model = next((g["model"] for g in groups.values() if g["model"]), "unknown")
    payload = {"model": model, "targets": out_targets}
    dst = BLOG_DATA / "layer1.json"
    dst.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    n_arms = sum(len(t["arms"]) for t in out_targets.values())
    print(f"  wrote {dst.relative_to(ROOT.parent.parent)} "
          f"({dst.stat().st_size / 1024:.0f} KB, {len(out_targets)} targets, "
          f"{n_arms} arm-curves, model={model})")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer1-only", action="store_true",
                    help="re-derive only layer1.json (skip the Layer-0 copy)")
    args = ap.parse_args()

    BLOG_DATA.mkdir(parents=True, exist_ok=True)
    print(f"bundling arena data -> {BLOG_DATA}")
    if not args.layer1_only:
        copy_layer0()
    build_layer1()
    print("done.")


if __name__ == "__main__":
    main()
