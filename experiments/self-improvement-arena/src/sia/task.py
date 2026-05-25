"""The symbolic-regression task: a hidden target function sampled into (x, y)
data, plus a held-out set for the success check. Fixed across all methods and
both layers -- only the proposer changes.

Each benchmark target is given BOTH as a numpy function (to generate data) and
as the exact grammar expression (to prove reachability and to sanity-check the
verifier). Targets are ordered by complexity; medium/harder are where greedy RL
is expected to collapse onto a simpler-but-wrong attractor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expression import GRAMMAR, KOZA_GRAMMAR, Grammar, Node, leaf


def _b(op: str, a: Node, c: Node) -> Node:
    return Node(op, [a, c])


def _u(op: str, a: Node) -> Node:
    return Node(op, [a])


_X = leaf("x")

# Exact target expressions, all reachable from GRAMMAR (vars {x}, consts
# {1, 2, 0.5}, unary {sin, cos}, binary {+,-,*,/}).
TARGET_EXPRS: dict[str, Node] = {
    # x*x + 1
    "easy": _b("+", _b("*", _X, _X), leaf("1.0")),
    # x*x + sin(x)   (main running example)
    "medium": _b("+", _b("*", _X, _X), _u("sin", _X)),
    # x*x*x - x + cos(2*x)
    "harder": _b(
        "+",
        _b("-", _b("*", _X, _b("*", _X, _X)), _X),
        _u("cos", _b("*", leaf("2.0"), _X)),
    ),
}

TARGET_FNS = {
    "easy": lambda x: x * x + 1.0,
    "medium": lambda x: x * x + np.sin(x),
    "harder": lambda x: x ** 3 - x + np.cos(2.0 * x),
}

# --- Nguyen benchmark suite (single-variable, Nguyen-1..8) -------------------
# DSR's standard Nguyen targets, searched over the Koza library
# {+,-,*,/,sin,cos,exp,log,x} (KOZA_GRAMMAR) -- NO constant terminals (constants
# are constructed, e.g. 1 = x/x). Per-benchmark sampling ranges from the DSO
# benchmarks spec. Nguyen-9..12 (two-variable) are deferred (need a 2nd variable).
NGUYEN_FNS = {
    "nguyen-1": lambda x: x**3 + x**2 + x,
    "nguyen-2": lambda x: x**4 + x**3 + x**2 + x,
    "nguyen-3": lambda x: x**5 + x**4 + x**3 + x**2 + x,
    "nguyen-4": lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x,
    "nguyen-5": lambda x: np.sin(x**2) * np.cos(x) - 1.0,
    "nguyen-6": lambda x: np.sin(x) + np.sin(x + x**2),
    "nguyen-7": lambda x: np.log(x + 1.0) + np.log(x**2 + 1.0),
    "nguyen-8": lambda x: np.sqrt(x),
}
# (lo, hi) sampling range per benchmark (DSO: U[-1,1] for 1-6, U[0,2] for 7, U[0,4] for 8)
NGUYEN_RANGES = {
    "nguyen-1": (-1.0, 1.0), "nguyen-2": (-1.0, 1.0), "nguyen-3": (-1.0, 1.0),
    "nguyen-4": (-1.0, 1.0), "nguyen-5": (-1.0, 1.0), "nguyen-6": (-1.0, 1.0),
    "nguyen-7": (0.0, 2.0), "nguyen-8": (0.0, 4.0),
}
# SymPy-parseable target strings, for the strict (exact-symbolic) recovery check.
TARGET_SYMPY = {
    "easy": "x**2 + 1",
    "medium": "x**2 + sin(x)",
    "harder": "x**3 - x + cos(2*x)",
}
NGUYEN_SYMPY = {
    "nguyen-1": "x**3 + x**2 + x",
    "nguyen-2": "x**4 + x**3 + x**2 + x",
    "nguyen-3": "x**5 + x**4 + x**3 + x**2 + x",
    "nguyen-4": "x**6 + x**5 + x**4 + x**3 + x**2 + x",
    "nguyen-5": "sin(x**2)*cos(x) - 1",
    "nguyen-6": "sin(x) + sin(x + x**2)",
    "nguyen-7": "log(x + 1) + log(x**2 + 1)",
    "nguyen-8": "sqrt(x)",
}
# Human-readable formulas for display (Nguyen targets have no exact grammar tree).
NGUYEN_FORMULAS = {
    "nguyen-1": "x^3 + x^2 + x",
    "nguyen-2": "x^4 + x^3 + x^2 + x",
    "nguyen-3": "x^5 + x^4 + x^3 + x^2 + x",
    "nguyen-4": "x^6 + x^5 + x^4 + x^3 + x^2 + x",
    "nguyen-5": "sin(x^2)*cos(x) - 1",
    "nguyen-6": "sin(x) + sin(x + x^2)",
    "nguyen-7": "log(x+1) + log(x^2+1)",
    "nguyen-8": "sqrt(x)",
}


@dataclass
class Task:
    name: str
    grammar: Grammar
    x_train: np.ndarray
    y_train: np.ndarray
    x_heldout: np.ndarray
    y_heldout: np.ndarray
    target_expr: Node
    target_sympy: str = ""   # SymPy-parseable target, for exact-symbolic recovery


def make_task(name: str, n_points: int = 30, x_range: tuple = (-3.0, 3.0),
              seed: int = 0) -> Task:
    """Build a task. Names in TARGET_FNS use the base grammar (our 3 targets);
    names in NGUYEN_FNS use the Koza grammar + their per-benchmark range."""
    if name in NGUYEN_FNS:                       # Nguyen suite -> Koza grammar
        fn = NGUYEN_FNS[name]
        lo, hi = NGUYEN_RANGES[name]
        grammar, target_expr = KOZA_GRAMMAR, None
        target_sympy = NGUYEN_SYMPY[name]
    elif name in TARGET_FNS:                     # our base 3 targets
        fn = TARGET_FNS[name]
        lo, hi = x_range
        grammar, target_expr = GRAMMAR, TARGET_EXPRS[name]
        target_sympy = TARGET_SYMPY[name]
    else:
        raise ValueError(f"unknown task {name!r}; choices: "
                         f"{list(TARGET_FNS) + list(NGUYEN_FNS)}")
    rng = np.random.default_rng(seed)
    x_train = np.sort(rng.uniform(lo, hi, size=n_points))
    # Held-out points are independently sampled in the same range, so the success
    # check rewards generalization rather than memorizing the training x's.
    x_heldout = np.sort(rng.uniform(lo, hi, size=n_points))
    return Task(
        name=name, grammar=grammar,
        x_train=x_train, y_train=fn(x_train),
        x_heldout=x_heldout, y_heldout=fn(x_heldout),
        target_expr=target_expr, target_sympy=target_sympy,
    )
