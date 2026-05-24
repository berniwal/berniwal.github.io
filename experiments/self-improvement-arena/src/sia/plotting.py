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
    "cvar": ("#e6ab02", "Risk-averse CVaR"),
    "greedy": ("#d95f02", "Greedy RL"),
    "risk_entropic": ("#e7298a", "Entropic RL (J_beta)"),
    "risk": ("#7570b3", "Risk-seeking RL (DSR)"),
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


def _summary_table_md(logs) -> str:
    """Final-budget summary: success rate, median evals-to-solve, mean best reward."""
    g = _group(logs)
    methods, targets = _methods_targets(logs)
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
    return "\n".join(lines) + "\n"


def _checkpoints(maxB: int) -> list[int]:
    fracs = [1 / 20, 1 / 10, 1 / 4, 1 / 2, 1.0]
    return sorted({int(maxB * f) for f in fracs if maxB * f >= 1})


def _scaling_table_md(logs) -> str:
    """Success rate at a few budget checkpoints -> the budget x method x complexity
    interaction at a glance."""
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
    return "\n".join(lines) + "\n"


def write_results_md(logs, out_dir: Path, reward_mode: str = "mse") -> str:
    """One self-contained results page: embeds the three figures and both tables, so
    the whole Layer-0 result reads top-to-bottom from a single file."""
    n_seeds = len({lg.seed for lg in logs})
    budget = logs[0].budget
    text = (
        f"# Layer 0 results — reward = `{reward_mode}`\n\n"
        f"Same task, same verifier, same budget ({_fmt_calls(budget)} verifier calls); "
        f"only the proposer differs. Mean over {n_seeds} seeds.\n\n"
        "## Best-so-far reward vs. budget\n\n"
        "![best-so-far reward vs. budget](curves.png)\n\n"
        "## Batch diversity — the collapse, visualized\n\n"
        "![batch diversity](diversity.png)\n\n"
        "## Success rate vs. budget\n\n"
        "![success rate vs. budget](scaling.png)\n\n"
        "## Summary (at full budget)\n\n"
        + _summary_table_md(logs)
        + "\n## Success rate at increasing verifier-call budgets\n\n"
        + _scaling_table_md(logs)
    )
    (out_dir / "results.md").write_text(text)
    return text


def _fmt_calls(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n / 1_000:g}k"
    return str(n)


def make_layer1(logs, out_dir: Path) -> str:
    """Layer-1 figures + table. Methods are arbitrary (model+temperature), so colors
    are auto-assigned rather than taken from the Layer-0 STYLE map."""
    out_dir.mkdir(parents=True, exist_ok=True)
    g = defaultdict(list)
    for lg in logs:
        g[(lg.target, lg.method)].append(lg)
    methods = sorted({lg.method for lg in logs})
    targets, seen = [], set()
    for t in ("easy", "medium", "harder"):
        if any(lg.target == t for lg in logs):
            targets.append(t)
            seen.add(t)
    for lg in logs:  # any non-standard targets, in encounter order
        if lg.target not in seen:
            targets.append(lg.target)
            seen.add(lg.target)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color = {m: cycle[i % len(cycle)] for i, m in enumerate(methods)}
    budget = max(lg.budget for lg in logs)
    grid = np.linspace(1, budget, 80)

    # best-so-far reward vs. generations
    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.2),
                             squeeze=False)
    for ax, t in zip(axes[0], targets):
        for m in methods:
            runs = g.get((t, m))
            if not runs:
                continue
            mean, std = aggregate(runs, grid, best_curve)
            ax.plot(grid, mean, color=color[m], label=m, lw=2)
            ax.fill_between(grid, mean - std, mean + std, color=color[m], alpha=0.15)
        ax.set_title(t)
        ax.set_xlabel("verifier calls (LLM generations)")
        ax.set_ylabel("best-so-far reward")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Layer 1 — LLM evolution: best reward vs. generations")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=130)
    plt.close(fig)

    # final best-reward bars per method per target
    x = np.arange(len(targets))
    w = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(2.4 * len(targets) + 3, 4.2))
    for i, m in enumerate(methods):
        vals = [float(np.mean([lg.best_reward[-1] for lg in g.get((t, m), [])]))
                if g.get((t, m)) else 0.0 for t in targets]
        ax.bar(x + i * w, vals, w, color=color[m], label=m)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(targets)
    ax.set_ylabel("mean final best reward")
    ax.set_title("Layer 1 — final best reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=130)
    plt.close(fig)

    # table
    lines = ["| target | arm | success rate | mean best reward | mean valid frac |",
             "|---|---|---|---|---|"]
    for t in targets:
        for m in methods:
            runs = g.get((t, m), [])
            if not runs:
                continue
            sr = success_rate(runs)
            mb = float(np.mean([lg.best_reward[-1] for lg in runs]))
            # Layer 1 stores per-batch valid fraction in the policy_entropy field
            vf = float(np.mean([np.nanmean(lg.policy_entropy) for lg in runs]))
            lines.append(f"| {t} | {m} | {sr:.2f} | {mb:.4f} | {vf:.2f} |")
    text = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(text)
    return text


def make_all(logs, out_dir: Path, reward_mode: str = "mse") -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(logs, out_dir / "curves.png", kind="reward")
    plot_curves(logs, out_dir / "diversity.png", kind="diversity")
    plot_budget_scaling(logs, out_dir / "scaling.png")
    return write_results_md(logs, out_dir, reward_mode=reward_mode)
