"""BFGS constant-fitting for LLM proposals (Layer 1).

An LLM doing symbolic regression proposes *parametric* expressions -- a structure
plus coefficients (`C*x*x + C*sin(x) + C`, or literal numbers). DSR's c-variants fit
those constants with a nonlinear optimizer before scoring (Petersen et al. 2021,
Sec. "Constant optimization"). We do the same: every constant placeholder `C` and
every numeric literal becomes a free parameter, fit to the training data, so the
reward reflects the *structure's best achievable fit* rather than the model's guessed
coefficients. This is the LLM analogue of the structure-vs-coefficients split.

scipy lives ONLY here, not in the numpy-only sia core (Layer 0 stays dependency-light).
"""
from __future__ import annotations

import re

import numpy as np
from scipy.optimize import minimize

from sia.expression import Node, evaluate, parse_expression

_PLACEHOLDER = re.compile(r"\bC\b|\bconst\b")


def _numeric_leaves(node: Node) -> list[Node]:
    """Every leaf that is a numeric constant (the variable x is left alone)."""
    out: list[Node] = []

    def walk(n: Node) -> None:
        if not n.children:
            if n.op != "x":
                try:
                    float(n.op)
                    out.append(n)
                except ValueError:
                    pass
        else:
            for c in n.children:
                walk(c)

    walk(node)
    return out


def fit_constants(node: Node, x: np.ndarray, y: np.ndarray, max_consts: int = 12) -> Node:
    """Fit every numeric-constant leaf to minimize MSE on (x, y), in place. No-op if
    there are no constants or implausibly many (we cap to keep the inner loop cheap)."""
    leaves = _numeric_leaves(node)
    if not leaves or len(leaves) > max_consts:
        return node
    init = np.array([float(leaf.op) for leaf in leaves])

    def loss(params: np.ndarray) -> float:
        for leaf, v in zip(leaves, params):
            leaf.op = repr(float(v))
        with np.errstate(all="ignore"):
            yp = evaluate(node, x)
        if not np.all(np.isfinite(yp)):
            return 1e12
        d = yp - y
        return float(np.mean(d * d))

    base = loss(init)
    res = minimize(loss, init, method="Nelder-Mead",
                   options={"maxiter": 200 * len(leaves), "xatol": 1e-7, "fatol": 1e-12})
    best = res.x if res.fun < base else init    # keep the init if the optimizer made it worse
    for leaf, v in zip(leaves, best):
        leaf.op = repr(float(v))
    return node


def parse_and_fit(text: str, x: np.ndarray, y: np.ndarray) -> Node | None:
    """Parse an LLM reply (with C placeholders) and BFGS-fit its constants. The `C`
    symbol becomes an initial value of 1.0; all constants are then fit to (x, y)."""
    node = parse_expression(_PLACEHOLDER.sub("1.0", text))
    if node is None:
        return None
    return fit_constants(node, x, y)
