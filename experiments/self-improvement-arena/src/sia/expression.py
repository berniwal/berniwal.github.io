"""Expression trees, grammar, evaluation, and the genetic-programming operators.

This module is shared by every method and (later) by Layer 1: it defines the
candidate space (the grammar) and how a candidate is turned into predictions.
Expressions are trees of :class:`Node`. The same trees serialize to/from prefix
(Polish) token sequences, which is the form the RNN policy emits.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

# --- Fixed grammar -----------------------------------------------------------
# Token vocabulary, in a fixed order so the RNN policy has a stable index space.
BINARY = ("+", "-", "*", "/")
UNARY = ("sin", "cos")
VARS = ("x",)
CONSTS = {"1.0": 1.0, "2.0": 2.0, "0.5": 0.5}  # token -> value

TERMINALS = tuple(VARS) + tuple(CONSTS)
TOKENS = tuple(BINARY) + tuple(UNARY) + tuple(VARS) + tuple(CONSTS)
ARITY = {
    **{op: 2 for op in BINARY},
    **{op: 1 for op in UNARY},
    **{t: 0 for t in TERMINALS},
}


@dataclass(frozen=True)
class Grammar:
    """The candidate space. Default instance :data:`GRAMMAR` is used everywhere."""

    binary: tuple = BINARY
    unary: tuple = UNARY
    terminals: tuple = TERMINALS
    tokens: tuple = TOKENS

    @property
    def operators(self) -> tuple:
        return self.binary + self.unary


GRAMMAR = Grammar()


@dataclass
class Node:
    """A single grammar token plus its children (empty for terminals)."""

    op: str
    children: list = field(default_factory=list)


def leaf(op: str) -> Node:
    return Node(op, [])


# --- Evaluation --------------------------------------------------------------
def evaluate(node: Node, x: np.ndarray) -> np.ndarray:
    """Vectorized evaluation. May return non-finite values (inf/NaN) for invalid
    expressions (div-by-zero, overflow); the verifier is responsible for checking
    finiteness. Callers should wrap this in ``np.errstate(all="ignore")``."""
    op = node.op
    if op == "x":
        return x
    if op in CONSTS:
        return np.full_like(x, CONSTS[op], dtype=float)
    if op == "+":
        return evaluate(node.children[0], x) + evaluate(node.children[1], x)
    if op == "-":
        return evaluate(node.children[0], x) - evaluate(node.children[1], x)
    if op == "*":
        return evaluate(node.children[0], x) * evaluate(node.children[1], x)
    if op == "/":
        return np.divide(evaluate(node.children[0], x), evaluate(node.children[1], x))
    if op == "sin":
        return np.sin(evaluate(node.children[0], x))
    if op == "cos":
        return np.cos(evaluate(node.children[0], x))
    raise ValueError(f"unknown op: {op!r}")


def complexity(node: Node) -> int:
    """Number of nodes in the tree (used for the length penalty)."""
    return 1 + sum(complexity(c) for c in node.children)


def to_infix(node: Node) -> str:
    """Human-readable string (for logs / the results table)."""
    op = node.op
    if ARITY[op] == 0:
        return op
    if op in UNARY:
        return f"{op}({to_infix(node.children[0])})"
    a, b = node.children
    return f"({to_infix(a)} {op} {to_infix(b)})"


# --- Prefix (Polish) serialization ------------------------------------------
def to_prefix(node: Node) -> list[str]:
    out = [node.op]
    for c in node.children:
        out.extend(to_prefix(c))
    return out


def from_prefix(tokens: list[str]) -> Node | None:
    """Parse a prefix token sequence into a tree. Returns None if the sequence is
    not a single complete, well-formed expression (too short or leftover tokens)."""
    it = iter(tokens)

    def build() -> Node:
        op = next(it)  # raises StopIteration if truncated
        n = ARITY[op]
        return Node(op, [build() for _ in range(n)])

    try:
        root = build()
    except (StopIteration, KeyError):
        return None
    if any(True for _ in it):  # leftover tokens -> not a single expression
        return None
    return root


# --- Random trees and GP operators ------------------------------------------
def random_tree(grammar: Grammar, rng: np.random.Generator, max_depth: int = 4) -> Node:
    """Grow a random tree. At ``max_depth`` (or with some probability) emit a
    terminal, otherwise an operator with recursively grown children."""
    if max_depth <= 1 or rng.random() < 0.35:
        return leaf(grammar.terminals[rng.integers(len(grammar.terminals))])
    op = grammar.operators[rng.integers(len(grammar.operators))]
    return Node(op, [random_tree(grammar, rng, max_depth - 1) for _ in range(ARITY[op])])


def _all_nodes(root: Node) -> list[Node]:
    nodes: list[Node] = []

    def walk(n: Node) -> None:
        nodes.append(n)
        for c in n.children:
            walk(c)

    walk(root)
    return nodes


def mutate_subtree(node: Node, grammar: Grammar, rng: np.random.Generator,
                   max_depth: int = 4) -> Node:
    """Replace a uniformly chosen subtree with a fresh random one."""
    out = copy.deepcopy(node)
    target = _all_nodes(out)[rng.integers(len(_all_nodes(out)))]
    repl = random_tree(grammar, rng, max_depth)
    target.op, target.children = repl.op, repl.children
    return out


def crossover_subtree(a: Node, b: Node, rng: np.random.Generator) -> Node:
    """Return a copy of ``a`` with one subtree replaced by a subtree from ``b``."""
    out = copy.deepcopy(a)
    donor = copy.deepcopy(b)
    target = _all_nodes(out)[rng.integers(len(_all_nodes(out)))]
    source = _all_nodes(donor)[rng.integers(len(_all_nodes(donor)))]
    target.op, target.children = source.op, source.children
    return out
