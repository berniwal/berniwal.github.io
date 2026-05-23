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

from .expression import GRAMMAR, Grammar, Node, leaf


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


@dataclass
class Task:
    name: str
    grammar: Grammar
    x_train: np.ndarray
    y_train: np.ndarray
    x_heldout: np.ndarray
    y_heldout: np.ndarray
    target_expr: Node


def make_task(name: str, n_points: int = 30, x_range: tuple = (-3.0, 3.0),
              seed: int = 0) -> Task:
    if name not in TARGET_FNS:
        raise ValueError(f"unknown task {name!r}; choices: {list(TARGET_FNS)}")
    rng = np.random.default_rng(seed)
    fn = TARGET_FNS[name]
    lo, hi = x_range
    x_train = np.sort(rng.uniform(lo, hi, size=n_points))
    # Held-out points are independently sampled in the same range, so the success
    # check rewards generalization rather than memorizing the training x's.
    x_heldout = np.sort(rng.uniform(lo, hi, size=n_points))
    return Task(
        name=name,
        grammar=GRAMMAR,
        x_train=x_train,
        y_train=fn(x_train),
        x_heldout=x_heldout,
        y_heldout=fn(x_heldout),
        target_expr=TARGET_EXPRS[name],
    )
