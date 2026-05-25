"""Expression trees, grammar, evaluation, and the genetic-programming operators.

This module is shared by every method and (later) by Layer 1: it defines the
candidate space (the grammar) and how a candidate is turned into predictions.
Expressions are trees of :class:`Node`. The same trees serialize to/from prefix
(Polish) token sequences, which is the form the RNN policy emits.
"""
from __future__ import annotations

import ast
import copy
import math
import re
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

# --- Extended (Koza / Nguyen) operators -------------------------------------
# Used ONLY when the Koza grammar is selected (a toggle). exp/log let the standard
# Nguyen suite be searched with DSR's library {+,-,*,/,sin,cos,exp,log,x}; there
# are NO constant terminals (constants are constructed, e.g. 1 = x/x), matching
# DSR's no-const standard Nguyen setup. The base grammar/TOKENS above are left
# untouched, so existing Layer-0 runs reproduce byte-for-byte.
TRIG = ("sin", "cos")                     # for the no-nested-trig constraint
EXTRA_UNARY = ("exp", "log")
INVERSE = {"exp": "log", "log": "exp"}    # for the no-inverse-of-unary constraint
ARITY.update({op: 1 for op in EXTRA_UNARY})


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

# The Koza/Nguyen grammar (toggle): DSR's ℒ₀ = {+,-,*,/,sin,cos,exp,log,x}, no
# constant terminals. Select it via make_task(..., suite="nguyen").
KOZA_UNARY = UNARY + EXTRA_UNARY
KOZA_TOKENS = tuple(BINARY) + tuple(KOZA_UNARY) + tuple(VARS)
KOZA_GRAMMAR = Grammar(binary=BINARY, unary=KOZA_UNARY, terminals=VARS,
                       tokens=KOZA_TOKENS)


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
    finiteness. Callers should wrap this in ``np.errstate(all="ignore")``.

    Leaf ops are either the variable ``"x"`` or a numeric constant. Grammar
    constants (``"1.0"`` etc.) and *arbitrary* numeric constants (e.g. ``"3.14"``
    produced by an LLM in Layer 1) are both handled, so the SAME verifier scores
    both Layer 0 trees and parsed LLM expressions."""
    op = node.op
    if not node.children:  # leaf: variable or numeric constant
        if op == "x":
            return x
        return np.full_like(x, float(op), dtype=float)
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
    if op == "exp":  # may overflow -> inf; verifier treats non-finite as invalid
        return np.exp(evaluate(node.children[0], x))
    if op == "log":  # log of <=0 -> nan/-inf -> invalid; that's the protection
        return np.log(evaluate(node.children[0], x))
    raise ValueError(f"unknown op: {op!r}")


def complexity(node: Node) -> int:
    """Number of nodes in the tree (used for the length penalty)."""
    return 1 + sum(complexity(c) for c in node.children)


def to_infix(node: Node) -> str:
    """Human-readable string (for logs / the results table). Children-based so it
    also handles arbitrary numeric constants from Layer 1 (not in ARITY)."""
    if not node.children:
        return node.op
    if len(node.children) == 1:
        return f"{node.op}({to_infix(node.children[0])})"
    a, b = node.children
    return f"({to_infix(a)} {node.op} {to_infix(b)})"


# --- SymPy bridge (DSR-style exact symbolic recovery) -----------------------
# DSR judges "recovery" by EXACT symbolic equivalence to the target (via a CAS),
# not by a numeric error threshold. On a bounded interval a smooth non-target
# (e.g. exp/trig combo) can fit a polynomial to MSE < 1e-6 yet be a different
# function -- so numeric "solve" overcounts. These helpers give the strict check.
def to_sympy(node: Node):
    """Convert a Node tree to a SymPy expression (variable ``x``). Numeric leaves
    become exact Rationals so simplification is not defeated by float noise."""
    import sympy as sp

    x = sp.Symbol("x")

    def conv(n: Node):
        if not n.children:
            return x if n.op == "x" else sp.Rational(n.op)
        a = conv(n.children[0])
        if n.op == "+":
            return a + conv(n.children[1])
        if n.op == "-":
            return a - conv(n.children[1])
        if n.op == "*":
            return a * conv(n.children[1])
        if n.op == "/":
            return a / conv(n.children[1])
        if n.op == "sin":
            return sp.sin(a)
        if n.op == "cos":
            return sp.cos(a)
        if n.op == "exp":
            return sp.exp(a)
        if n.op == "log":
            return sp.log(a)
        raise ValueError(f"to_sympy: unknown op {n.op!r}")

    return conv(node)


def sympy_equivalent(node: Node, target_str: str) -> bool:
    """True iff ``node`` is exactly symbolically equivalent to ``target_str``
    (DSR's recovery criterion). Robust to CAS hiccups (a bad expression -> False),
    but a MISSING sympy raises loudly: sympy is a required dependency, and silently
    returning False would mis-report every run as 0% recovery (it once did, on a pod
    where sympy wasn't installed)."""
    import sympy as sp  # required dep -- let ImportError propagate, do NOT swallow it

    try:
        expr = to_sympy(node)
        target = sp.sympify(target_str)
        return bool(sp.simplify(expr - target) == 0)
    except Exception:
        return False


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


# --- Parsing free-form infix expressions (Layer 1: LLM text -> Node) ---------
# An LLM emits expressions as text ("x*x + sin(x)"). We parse them into the SAME
# Node representation Layer 0 uses, so the SAME verifier scores them. Anything
# outside the grammar (unknown functions, non-integer powers, ...) returns None
# -> the verifier treats it as an invalid candidate (reward 0).
_AST_BINOP = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_AST_FUNCS = {"sin", "cos"}
_NAMED_CONSTS = {"pi": math.pi, "e": math.e}
_MAX_POW = 6  # x**n is expanded to repeated multiplication for 1 <= n <= _MAX_POW


def _from_ast(node: ast.AST) -> Node | None:
    if isinstance(node, ast.Expression):
        return _from_ast(node.body)
    if isinstance(node, ast.BinOp):
        opt = type(node.op)
        if opt in _AST_BINOP:
            lhs, rhs = _from_ast(node.left), _from_ast(node.right)
            return None if lhs is None or rhs is None else Node(_AST_BINOP[opt], [lhs, rhs])
        if opt is ast.Pow:  # x**n -> x*x*...*x  (small integer n only)
            base, exp = _from_ast(node.left), node.right
            if base is None or not isinstance(exp, ast.Constant):
                return None
            v = exp.value
            if isinstance(v, (int, float)) and float(v).is_integer() and 1 <= int(v) <= _MAX_POW:
                out = copy.deepcopy(base)
                for _ in range(int(v) - 1):
                    out = Node("*", [out, copy.deepcopy(base)])
                return out
            return None
        return None
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            return _from_ast(node.operand)
        if isinstance(node.op, ast.USub):
            v = _from_ast(node.operand)
            return None if v is None else Node("-", [leaf("0.0"), v])
        return None
    if isinstance(node, ast.Call):
        if (isinstance(node.func, ast.Name) and node.func.id in _AST_FUNCS
                and len(node.args) == 1 and not node.keywords):
            a = _from_ast(node.args[0])
            return None if a is None else Node(node.func.id, [a])
        return None
    if isinstance(node, ast.Name):
        if node.id == "x":
            return leaf("x")
        if node.id in _NAMED_CONSTS:
            return leaf(repr(float(_NAMED_CONSTS[node.id])))
        return None
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return leaf(repr(float(node.value)))
        return None
    return None


def _candidate_strings(text: str):
    """Yield cleaned expression candidates from an LLM reply, in priority order:
    each line (and the whole text), plus the part after a trailing ``=``/``:``,
    so prose like 'The expression is: x*x' or 'y = x*x' is handled."""
    s = text.replace("```python", " ").replace("```", " ")
    seen = set()
    for line in s.splitlines() + [s]:
        pieces = [line]
        if "=" in line:
            pieces.append(line.split("=")[-1])
        if ":" in line:
            pieces.append(line.split(":")[-1])
        for p in pieces:
            c = p.replace("^", "**").strip().rstrip(".,;: ").strip()
            if c and c not in seen:
                seen.add(c)
                yield c


def _consts_to_one(node: Node) -> None:
    """Set every numeric-constant leaf to 1.0 in place (the variable x is left)."""
    if not node.children:
        if node.op != "x":
            node.op = "1.0"
    else:
        for c in node.children:
            _consts_to_one(c)


def parse_expression(text: str, const_placeholder: bool = False) -> Node | None:
    """Parse an LLM's free-form reply into a Node, or None if no candidate is a
    valid in-grammar expression. Tries each line/segment and returns the first
    that parses (so leading prose is tolerated).

    ``const_placeholder``: DSR-style constant token. The bare symbol ``C`` (or
    ``const``) is treated as a placeholder and, together with any numeric literal,
    is set to 1.0 — so the model proposes *structure* (``C*x*x + C*sin(x)``) and we
    score it with all constants = 1 (a stand-in until BFGS constant-fitting)."""
    if const_placeholder:
        text = re.sub(r"\bC\b|\bconst\b", "1", text)
    for cand in _candidate_strings(text):
        try:
            tree = ast.parse(cand, mode="eval")
        except (SyntaxError, ValueError):
            continue
        node = _from_ast(tree)
        if node is not None:
            if const_placeholder:
                _consts_to_one(node)
            return node
    return None
