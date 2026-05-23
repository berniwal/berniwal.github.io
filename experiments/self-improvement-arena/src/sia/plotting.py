"""Figures and the results table for Layer 0."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import (RunLog, aggregate, best_curve, best_reward_at,  # noqa: E402
                      diversity_curve, median_evals_to_solve, success_rate,
                      success_rate_at)

# Stable color/label per method (keys must match config method names).
STYLE = {
    "random": ("#9e9e9e", "Random search"),
    "gp": ("#1b9e77", "Evolution (GP)"),
    "greedy": ("#d95f02", "Greedy RL"),
    "risk": ("#7570b3", "Risk-seeking RL (DSR)"),
    "risk_entropic": ("#e7298a", "Entropic RL (J_beta)"),
}


def _group(logs: list[RunLog]) -> dict:
    g: dict = defaultdict(list)
    for lg in logs:
        g[(lg.target, lg.method)].append(lg)
    return g


def _methods_targets(logs: list[RunLog]):
    methods = [m for m in STYLE if any(lg.method == m for lg in logs)]
    targets, seen = [], set()
    for lg in logs:  # preserve config order of first appearance
        if lg.target not in seen:
            targets.append(lg.target)
            seen.add(lg.target)
    return methods, targets


def plot_curves(logs, out: Path, kind: str = "reward") -> None:
    g = _group(logs)
    methods, targets = _methods_targets(logs)
    budget = logs[0].budget
    grid = np.linspace(budget * 0.0, budget, 120)[1:]
    curve_fn = best_curve if kind == "reward" else diversity_curve
    ylabel = "best-so-far reward" if kind == "reward" else "batch diversity (unique fraction)"

    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.2),
                             squeeze=False)
    for ax, target in zip(axes[0], targets):
        for m in methods:
            runs = g.get((target, m))
            if not runs:
                continue
            color, label = STYLE[m]
            mean, std = aggregate(runs, grid, curve_fn)
            ax.plot(grid, mean, color=color, label=label, lw=2)
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(target)
        ax.set_xlabel("verifier calls")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{ylabel} vs. budget  (mean +/- std over {len(next(iter(g.values())))} seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_success(logs, out: Path) -> None:
    g = _group(logs)
    methods, targets = _methods_targets(logs)
    x = np.arange(len(targets))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(2.2 * len(targets) + 2, 4.2))
    for i, m in enumerate(methods):
        rates = [success_rate(g.get((t, m), [])) for t in targets]
        color, label = STYLE[m]
        ax.bar(x + i * w, rates, w, color=color, label=label)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(targets)
    ax.set_ylabel("success rate (exact recovery)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Success rate over {len(next(iter(g.values())))} seeds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_budget_scaling(logs, out: Path) -> None:
    """Success rate vs. budget (log x) per method per target -- the figure that
    shows which method wins under which budget/complexity, and whether the learned
    policy (DSR) overtakes evolution (GP) as the budget grows."""
    g = _group(logs)
    methods, targets = _methods_targets(logs)
    maxB = logs[0].budget
    grid = np.unique(np.geomspace(max(maxB // 500, 200), maxB, 50).astype(int))
    n_seeds = len(next(iter(g.values())))

    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.2),
                             squeeze=False)
    for ax, target in zip(axes[0], targets):
        for m in methods:
            runs = g.get((target, m))
            if not runs:
                continue
            color, label = STYLE[m]
            sr = [success_rate_at(runs, int(B)) for B in grid]
            ax.plot(grid, sr, color=color, label=label, lw=2, marker="o", ms=2.5)
        ax.set_xscale("log")
        ax.set_title(target)
        ax.set_xlabel("verifier-call budget (log scale)")
        ax.set_ylabel("success rate (exact recovery)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Success rate vs. budget  (over {n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def write_table(logs, out: Path) -> str:
    g = _group(logs)
    methods, targets = _methods_targets(logs)
    grid = np.array([logs[0].budget])
    lines = ["| target | method | success rate | median evals-to-solve | mean best reward |",
             "|---|---|---|---|---|"]
    for t in targets:
        for m in methods:
            runs = g.get((t, m), [])
            if not runs:
                continue
            sr = success_rate(runs)
            med = median_evals_to_solve(runs)
            mean_best = float(np.mean([lg.best_reward[-1] for lg in runs]))
            med_s = f"{int(med)}" if med is not None else "-"
            lines.append(f"| {t} | {STYLE[m][1]} | {sr:.2f} | {med_s} | {mean_best:.4f} |")
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    return text


def _checkpoints(maxB: int) -> list[int]:
    fracs = [1 / 20, 1 / 10, 1 / 4, 1 / 2, 1.0]
    return sorted({int(maxB * f) for f in fracs if maxB * f >= 1})


def write_scaling_table(logs, out: Path) -> str:
    """Success rate at a few budget checkpoints -> shows the budget x method x
    complexity interaction at a glance."""
    g = _group(logs)
    methods, targets = _methods_targets(logs)
    cps = _checkpoints(logs[0].budget)
    header = "| target | method | " + " | ".join(_fmt_calls(c) for c in cps) + " |"
    lines = [header, "|" + "---|" * (len(cps) + 2)]
    for t in targets:
        for m in methods:
            runs = g.get((t, m), [])
            if not runs:
                continue
            cells = " | ".join(f"{success_rate_at(runs, c):.2f}" for c in cps)
            lines.append(f"| {t} | {STYLE[m][1]} | {cells} |")
    text = ("Success rate (exact recovery) at increasing verifier-call budgets:\n\n"
            + "\n".join(lines) + "\n")
    out.write_text(text)
    return text


def _fmt_calls(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n / 1_000:g}k"
    return str(n)


def make_all(logs, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(logs, out_dir / "curves.png", kind="reward")
    plot_curves(logs, out_dir / "diversity.png", kind="diversity")
    plot_success(logs, out_dir / "success.png")
    plot_budget_scaling(logs, out_dir / "scaling.png")
    write_scaling_table(logs, out_dir / "scaling_table.md")
    return write_table(logs, out_dir / "table.md")
