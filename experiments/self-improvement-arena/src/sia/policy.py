"""A tiny autoregressive token policy in pure numpy: a vanilla (Elman) RNN that
emits expressions one prefix token at a time, with manual BPTT.

The whole point of writing this by hand is pedagogical: all three RL arms share
*exactly* this code. A policy-gradient step is always

    maximize  sum_i  w_i * sum_t log pi(a_t^i)   (+ entropy bonus)

and the arms differ ONLY in the per-trajectory weight vector ``w`` (greedy =
mean-baselined reward; risk-seeking = top-epsilon quantile; entropic = softmax
tilt). Backprop is identical -> it lives here, once.

Generation uses arity tracking with feasibility masking so every sampled token
sequence decodes to a complete, finite tree within ``max_length``.
"""
from __future__ import annotations

import numpy as np

from .expression import ARITY, TOKENS, Node, from_prefix

_ARITY_ARR = np.array([ARITY[t] for t in TOKENS])
_BINARY = _ARITY_ARR == 2
_UNARY = _ARITY_ARR == 1
V = len(TOKENS)


class _Adam:
    def __init__(self, params: dict, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class RNNPolicy:
    def __init__(self, hidden: int = 32, max_length: int = 24, lr: float = 0.01,
                 grad_clip: float = 5.0, seed: int = 0):
        self.H = hidden
        self.L = max_length
        self.grad_clip = grad_clip
        rng = np.random.default_rng(seed)
        s = 0.1
        self.p = {
            "Wxh": rng.normal(0, s, (V, hidden)),
            "Whh": rng.normal(0, s, (hidden, hidden)),
            "bh": np.zeros(hidden),
            "Who": rng.normal(0, s, (hidden, V)),
            "bo": np.zeros(V),
        }
        self.opt = _Adam(self.p, lr)
        self._cache = None
        self._last_entropy = 0.0

    # --- sampling (vectorized lockstep rollout over the batch) ----------------
    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Node]:
        B, H, L = batch_size, self.H, self.L
        h = np.zeros((B, H))
        x = np.zeros((B, V))            # BOS input
        counter = np.ones(B, dtype=int)  # open operand slots
        done = np.zeros(B, dtype=bool)
        toks = np.full((B, L), -1, dtype=int)
        length = np.zeros(B, dtype=int)
        cache = []  # per-step dicts for BPTT

        for t in range(L):
            active = ~done
            h_prev = h
            z = x @ self.p["Wxh"] + h_prev @ self.p["Whh"] + self.p["bh"]
            h = np.tanh(z)
            logits = h @ self.p["Who"] + self.p["bo"]

            slack = L - t - counter  # >=0 invariant; room left to close the tree
            add = np.zeros((B, V))
            add[(slack < 2)[:, None] & _BINARY[None, :]] = -np.inf
            add[(slack < 1)[:, None] & _UNARY[None, :]] = -np.inf
            logits = logits + add

            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)

            u = rng.random((B, 1))
            a = (np.cumsum(probs, axis=1) < u).sum(axis=1)
            a = np.clip(a, 0, V - 1)

            cache.append({"x": x, "h_prev": h_prev, "h": h, "probs": probs,
                          "a": a, "active": active.copy()})

            ar = _ARITY_ARR[a]
            counter = np.where(active, counter + ar - 1, counter)
            length += active.astype(int)
            toks[active, t] = a[active]
            done = done | (active & (counter == 0))
            x = np.zeros((B, V))
            x[np.arange(B), a] = 1.0
            if done.all():
                break

        self._cache = {"steps": cache, "B": B}
        # entropy of the realized step distributions (diversity diagnostic)
        ent = [(-(s["probs"] * np.log(s["probs"] + 1e-12)).sum(1) * s["active"]).sum()
               for s in cache]
        denom = sum(s["active"].sum() for s in cache)
        self._last_entropy = float(np.sum(ent) / max(denom, 1))

        trees = []
        for i in range(B):
            seq = [TOKENS[j] for j in toks[i, : length[i]]]
            node = from_prefix(seq)
            trees.append(node if node is not None else Node("x", []))
        return trees

    def last_entropy(self) -> float:
        return self._last_entropy

    # --- policy-gradient update (shared by all RL arms) ----------------------
    def reinforce(self, weights: np.ndarray, ent_coef: float = 0.0) -> None:
        """One gradient-ascent step on  sum_i w_i * sum_t log pi(a_t) + ent_coef*H.
        ``weights`` is the per-trajectory advantage; the arm decides how to set it."""
        c = self._cache
        steps, B = c["steps"], c["B"]
        w = weights.reshape(B, 1)
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        dh_next = np.zeros((B, self.H))

        for s in reversed(steps):
            probs, a, active = s["probs"], s["a"], s["active"][:, None]
            onehot = np.zeros((B, V))
            onehot[np.arange(B), a] = 1.0
            logp = np.log(probs + 1e-12)
            ent = -(probs * logp).sum(1, keepdims=True)
            # gradient of the (maximized) objective wrt logits
            d_obj = w * (onehot - probs) - ent_coef * probs * (logp + ent)
            # loss = -objective; mask inactive steps; average over batch
            dO = -(d_obj * active) / B

            g["Who"] += s["h"].T @ dO
            g["bo"] += dO.sum(0)
            dh = dO @ self.p["Who"].T + dh_next
            dz = dh * (1 - s["h"] ** 2)
            g["Wxh"] += s["x"].T @ dz
            g["Whh"] += s["h_prev"].T @ dz
            g["bh"] += dz.sum(0)
            dh_next = dz @ self.p["Whh"].T

        norm = np.sqrt(sum((gv ** 2).sum() for gv in g.values()))
        if norm > self.grad_clip:
            scale = self.grad_clip / (norm + 1e-12)
            g = {k: v * scale for k, v in g.items()}
        self.opt.step(self.p, g)
