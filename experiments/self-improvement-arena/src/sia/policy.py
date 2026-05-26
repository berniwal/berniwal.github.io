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

from .expression import ARITY, CONSTS, INVERSE, TOKENS, TRIG, Node, from_prefix

_NEG_INF = -np.inf


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
                 grad_clip: float = 5.0, seed: int = 0,
                 constraints: bool = False, min_length: int = 4,
                 entropy_gamma: float = 1.0, tokens: tuple = TOKENS,
                 obs_struct: bool = False, cell: str = "rnn"):
        self.H = hidden
        self.L = max_length
        self.grad_clip = grad_clip
        # Token vocabulary the policy emits over. Default is the base grammar, so
        # existing runs are byte-identical; pass a different token tuple (e.g. the
        # Koza/Nguyen set) to search a different space. All vocab-derived arrays are
        # instance-level so two policies can use different grammars in one process.
        self.tokens = tuple(tokens)
        self.V = len(self.tokens)
        self._arity = np.array([ARITY[t] for t in self.tokens])
        self._binary = self._arity == 2
        self._unary = self._arity == 1
        self._term = self._arity == 0
        self._trig = np.array([t in TRIG for t in self.tokens])    # sin/cos only
        self._const = np.array([t in CONSTS for t in self.tokens])  # constant leaves
        # inverse-token index per vocab position (exp<->log), -1 if none -> constraint #3
        idx = {t: i for i, t in enumerate(self.tokens)}
        self._inv = np.full(self.V, -1, dtype=int)
        for a, b in INVERSE.items():
            if a in idx and b in idx:
                self._inv[idx[a]] = idx[b]
        # DSR-style a-priori constraints on the sampled tree (Petersen et al. 2021,
        # Sec. 3.2). Off by default -> the original fast counter-based sampler and
        # the existing reproducible results. On -> the four constraints below.
        self.constraints = constraints
        self.min_length = min_length
        # Hierarchical-entropy discount (Landajuela et al. 2021): the entropy bonus at
        # token position t is weighted by entropy_gamma**t, so EARLY (structural) tokens
        # get the most exploration pressure -- counters "early commitment". gamma=1.0 is
        # the flat per-token entropy bonus (original DSR / our prior behavior).
        self.entropy_gamma = entropy_gamma
        # Structural observations (DSR Sec. 3.1): in addition to the previous token, feed
        # the open slot's PARENT operator and PREVIOUS SIBLING as RNN inputs at each step.
        # Without these, the policy can only see prev-token context, which makes targets
        # with long-range structural symmetry (e.g. Nguyen-7 = log(x+1) + log(x^2+1) -- a
        # right child whose shape mirrors the left) very hard to discover. Requires
        # constraints=True (we need the stack to read parent/sibling). Off by default so
        # existing runs are byte-identical; turn on for the DSR-faithful Nguyen sweep.
        self.obs_struct = obs_struct
        if obs_struct and not constraints:
            raise ValueError("obs_struct=True requires constraints=True (stack tracking)")
        if cell not in ("rnn", "lstm"):
            raise ValueError(f"cell must be 'rnn' or 'lstm', got {cell!r}")
        self.cell = cell
        rng = np.random.default_rng(seed)
        s = 0.1
        # Output projection (hidden -> logit) is shared between both cell types.
        self.p = {
            "Who": rng.normal(0, s, (hidden, self.V)),
            "bo":  np.zeros(self.V),
        }
        if cell == "rnn":
            # Elman/tanh recurrence (existing): one input embedding per source.
            self.p["Wxh"] = rng.normal(0, s, (self.V, hidden))
            self.p["Whh"] = rng.normal(0, s, (hidden, hidden))
            self.p["bh"]  = np.zeros(hidden)
            if obs_struct:
                self.p["Wph"] = rng.normal(0, s, (self.V + 1, hidden))
                self.p["Wsh"] = rng.normal(0, s, (self.V + 1, hidden))
                self.p["Wph"][self.V] = 0.0
                self.p["Wsh"][self.V] = 0.0
        else:  # cell == "lstm"
            # Standard LSTM (Hochreiter & Schmidhuber 1997). Per-step computation:
            #   pre = Wx[x] + Wh @ h_prev + Wp[parent] + Ws[sib] + b   shape (B, 4H)
            #   split pre into (i, f, o, g); apply sigmoid/tanh; update cell state.
            # We stack the 4 gate matrices into one (in_dim, 4H) per input source so the
            # forward is a single matmul; gradients are split per-gate in BPTT.
            H4 = 4 * hidden
            self.p["lstm_Wx"] = rng.normal(0, s, (self.V, H4))
            self.p["lstm_Wh"] = rng.normal(0, s, (hidden, H4))
            self.p["lstm_b"]  = np.zeros(H4)
            # Forget-gate bias = 1 (Jozefowicz et al. 2015) so cell state is preserved
            # by default, letting backprop discover when to forget rather than learning
            # "keep" from scratch. Gate order is [i, f, o, g].
            self.p["lstm_b"][hidden:2 * hidden] = 1.0
            if obs_struct:
                self.p["lstm_Wp"] = rng.normal(0, s, (self.V + 1, H4))
                self.p["lstm_Ws"] = rng.normal(0, s, (self.V + 1, H4))
                self.p["lstm_Wp"][self.V] = 0.0
                self.p["lstm_Ws"][self.V] = 0.0
        self._empty_idx = self.V
        self.opt = _Adam(self.p, lr)
        self._cache = None
        self._last_entropy = 0.0

    # --- sampling (vectorized lockstep rollout over the batch) ----------------
    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Node]:
        B, H, L, V = batch_size, self.H, self.L, self.V
        h = np.zeros((B, H))
        c = np.zeros((B, H)) if self.cell == "lstm" else None  # LSTM cell state
        x = np.zeros((B, V))            # BOS input
        done = np.zeros(B, dtype=bool)
        toks = np.full((B, L), -1, dtype=int)
        length = np.zeros(B, dtype=int)
        cache = []  # per-step dicts for BPTT

        # Per-sequence open-slot bookkeeping. Without constraints we only need a
        # count of open slots (vectorized). With constraints we keep an explicit
        # stack per sequence so each open slot carries its parent context (needed
        # for the nested-trig / all-constant / inverse rules); see _mask_row. A slot
        # is (under_trig, parent_op_idx, parent_record); root is (False, -1, None).
        counter = np.ones(B, dtype=int)
        stacks = ([[(False, -1, None)] for _ in range(B)]
                  if self.constraints else None)

        for t in range(L):
            active = ~done
            h_prev = h
            c_prev = c                  # None for RNN, (B, H) for LSTM
            # Gather parent + previous-sibling indices for each sequence's open slot.
            # Sentinel V means "empty" (root parent / no prior sibling); Wph[V] / Wsh[V]
            # are the (zero-initialized) empty-context embeddings.
            if self.obs_struct:
                parent_idx = np.full(B, self._empty_idx, dtype=int)
                sib_idx = np.full(B, self._empty_idx, dtype=int)
                for i in range(B):
                    if active[i] and stacks[i]:
                        _, p_op, rec = stacks[i][-1]
                        if p_op >= 0:
                            parent_idx[i] = p_op
                        if rec is not None and rec[2] != self._empty_idx:
                            sib_idx[i] = rec[2]
            if self.cell == "rnn":
                z = x @ self.p["Wxh"] + h_prev @ self.p["Whh"] + self.p["bh"]
                if self.obs_struct:
                    z = z + self.p["Wph"][parent_idx] + self.p["Wsh"][sib_idx]
                h = np.tanh(z)
                gates = None
            else:  # cell == "lstm"
                pre = x @ self.p["lstm_Wx"] + h_prev @ self.p["lstm_Wh"] + self.p["lstm_b"]
                if self.obs_struct:
                    pre = pre + self.p["lstm_Wp"][parent_idx] + self.p["lstm_Ws"][sib_idx]
                # Split into 4 gate pre-activations [i, f, o, g], then non-linearities.
                pi = 1.0 / (1.0 + np.exp(-pre[:, :H]))            # input gate
                pf = 1.0 / (1.0 + np.exp(-pre[:, H:2*H]))         # forget gate
                po = 1.0 / (1.0 + np.exp(-pre[:, 2*H:3*H]))       # output gate
                pg = np.tanh(pre[:, 3*H:])                         # candidate
                c = pf * c_prev + pi * pg
                ct = np.tanh(c)
                h = po * ct
                gates = (pi, pf, po, pg, c, ct)
            logits = h @ self.p["Who"] + self.p["bo"]

            add = np.zeros((B, V))
            if self.constraints:
                for i in range(B):
                    if active[i]:
                        add[i] = self._mask_row(stacks[i], length[i], t)
            else:
                slack = L - t - counter  # >=0 invariant; room left to close the tree
                add[(slack < 2)[:, None] & self._binary[None, :]] = _NEG_INF
                add[(slack < 1)[:, None] & self._unary[None, :]] = _NEG_INF
            logits = logits + add

            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)

            u = rng.random((B, 1))
            a = (np.cumsum(probs, axis=1) < u).sum(axis=1)
            a = np.clip(a, 0, V - 1)

            entry = {"x": x, "h_prev": h_prev, "h": h, "probs": probs,
                     "a": a, "active": active.copy()}
            if self.obs_struct:
                entry["parent_idx"] = parent_idx
                entry["sib_idx"] = sib_idx
            if self.cell == "lstm":
                entry["c_prev"] = c_prev
                entry["gates"] = gates  # (pi, pf, po, pg, c, ct)
            cache.append(entry)

            if self.constraints:
                for i in range(B):
                    if not active[i]:
                        continue
                    ai = int(a[i])
                    under, _pop, rec = stacks[i].pop()
                    if rec is not None:              # update parent's child tally
                        rec[0] -= 1
                        rec[1] = rec[1] and bool(self._const[ai])
                        # NEW: the just-emitted token becomes the previous-sibling for
                        # the NEXT child slot under the same parent (siblings share rec).
                        rec[2] = ai
                    ar = int(self._arity[ai])
                    if ar > 0:                       # push this op's child slots
                        # [remaining, all-constant-so-far, prev_sib_idx] (V = empty / no prior sibling)
                        child_rec = [ar, True, self._empty_idx]
                        child_under = under or bool(self._trig[ai])
                        stacks[i].extend([(child_under, ai, child_rec)] * ar)
                    length[i] += 1
                    toks[i, t] = ai
                    if not stacks[i]:
                        done[i] = True
            else:
                ar = self._arity[a]
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
            seq = [self.tokens[j] for j in toks[i, : length[i]]]
            node = from_prefix(seq)
            trees.append(node if node is not None else Node("x", []))
        return trees

    def _mask_row(self, stack: list, length_i: int, t: int) -> np.ndarray:
        """Additive logit mask (0 / -inf) for the next token of ONE sequence under
        the DSR a-priori constraints. ``stack[-1]`` is the open slot being filled,
        carrying ``(under_trig, parent_op_idx, parent_record)`` where
        ``parent_record`` is ``[children_remaining, all_children_constant_so_far]``.

        (1) length: forbid finishing the tree before ``min_length`` (and the shared
            slack rule keeps it finishable within ``max_length``).
        (2) not-all-constant: when filling an operator's last child and every prior
            child was a constant leaf, forbid constant leaves here.
        (3) no-inverse-of-unary: under a unary op with an inverse (exp/log), forbid
            that inverse as the child (no exp(log(.)) / log(exp(.))).
        (4) no-nested-trig: under a trig ancestor, forbid trig operators."""
        add = np.zeros(self.V)
        counter = len(stack)
        slack = self.L - t - counter
        if slack < 2:                       # not enough room to close a binary op
            add[self._binary] = _NEG_INF
        if slack < 1:                       # ... or a unary op
            add[self._unary] = _NEG_INF
        under, parent_op, rec = stack[-1]
        if under:                           # (4) descendants of trig are not trig
            add[self._trig] = _NEG_INF
        if rec is not None and rec[0] == 1 and rec[1]:  # (2) avoid all-constant operands
            add[self._const] = _NEG_INF
        if parent_op >= 0 and self._inv[parent_op] >= 0:  # (3) no inverse of parent unary
            add[self._inv[parent_op]] = _NEG_INF
        if counter == 1 and (length_i + 1) < self.min_length:  # (1) min length
            add[self._term] = _NEG_INF
        if np.all(np.isneginf(add)):        # safety: never mask everything
            add = np.zeros(self.V)
            if slack < 2:
                add[self._binary] = _NEG_INF
            if slack < 1:
                add[self._unary] = _NEG_INF
        return add

    def last_entropy(self) -> float:
        return self._last_entropy

    # --- policy-gradient update (shared by all RL arms) ----------------------
    def reinforce(self, weights: np.ndarray, ent_coef: float = 0.0) -> None:
        """One gradient-ascent step on  sum_i w_i * sum_t log pi(a_t) + ent_coef*H.
        ``weights`` is the per-trajectory advantage; the arm decides how to set it."""
        c = self._cache
        steps, B, V = c["steps"], c["B"], self.V
        w = weights.reshape(B, 1)
        # PG term is averaged over the SAMPLES THAT CONTRIBUTE (nonzero weight), the
        # entropy term over the whole batch -- DSR keeps these two normalizations
        # separate. For greedy/entropic every sample contributes (n_pg == B -> no
        # change); for the risk/cvar tail arms only ~eps*B do, so this restores DSR's
        # 1/(eps N) risk-gradient scale. Previously both used 1/N, making the risk
        # gradient ~eps (e.g. 20x at eps=0.05) too weak at the shared learning rate.
        n_pg = max(int(np.count_nonzero(weights)), 1)
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        H = self.H
        dh_next = np.zeros((B, H))
        # LSTM also has a cell-state gradient that flows from step t+1 back through f_{t+1}.
        dc_next = np.zeros((B, H)) if self.cell == "lstm" else None

        for t in range(len(steps) - 1, -1, -1):  # reverse order, but track position t
            s = steps[t]
            probs, a, active = s["probs"], s["a"], s["active"][:, None]
            onehot = np.zeros((B, V))
            onehot[np.arange(B), a] = 1.0
            logp = np.log(probs + 1e-12)
            ent = -(probs * logp).sum(1, keepdims=True)
            # gradient of the (maximized) objective wrt logits. The entropy bonus at
            # position t is discounted by entropy_gamma**t (hierarchical entropy):
            # gamma=1 -> flat (every token equal); gamma<1 -> early tokens weigh most.
            ent_w = ent_coef * (self.entropy_gamma ** t)
            # PG term averaged over contributing samples (n_pg); entropy over the batch (B)
            d_obj = (w * (onehot - probs)) / n_pg - (ent_w * probs * (logp + ent)) / B
            dO = -(d_obj * active)

            g["Who"] += s["h"].T @ dO
            g["bo"] += dO.sum(0)
            dh = dO @ self.p["Who"].T + dh_next

            if self.cell == "rnn":
                dz = dh * (1 - s["h"] ** 2)
                g["Wxh"] += s["x"].T @ dz
                g["Whh"] += s["h_prev"].T @ dz
                g["bh"] += dz.sum(0)
                if self.obs_struct:
                    # gradient through the parent + previous-sibling embeddings: each row
                    # of the gather contributes its dz back to the gathered index. np.add.at
                    # is the unbuffered scatter-add that handles repeated indices correctly.
                    np.add.at(g["Wph"], s["parent_idx"], dz)
                    np.add.at(g["Wsh"], s["sib_idx"], dz)
                dh_next = dz @ self.p["Whh"].T
            else:  # cell == "lstm"
                pi, pf, po, pg, c, ct = s["gates"]
                c_prev_t = s["c_prev"]
                # h = po * ct  ->  split gradient between output gate and cell-state-via-tanh
                do = dh * ct
                dct = dh * po
                # ct = tanh(c)  ->  local dc from this step plus dc inherited from t+1
                dc = dc_next + dct * (1.0 - ct * ct)
                # c = pf * c_prev + pi * pg
                df = dc * c_prev_t
                dc_prev = dc * pf
                di = dc * pg
                dg = dc * pi
                # backprop through the gate non-linearities
                dpi = di * pi * (1.0 - pi)
                dpf = df * pf * (1.0 - pf)
                dpo = do * po * (1.0 - po)
                dpg = dg * (1.0 - pg * pg)
                dpre = np.concatenate([dpi, dpf, dpo, dpg], axis=1)
                g["lstm_Wx"] += s["x"].T @ dpre
                g["lstm_Wh"] += s["h_prev"].T @ dpre
                g["lstm_b"]  += dpre.sum(0)
                if self.obs_struct:
                    np.add.at(g["lstm_Wp"], s["parent_idx"], dpre)
                    np.add.at(g["lstm_Ws"], s["sib_idx"], dpre)
                dh_next = dpre @ self.p["lstm_Wh"].T
                dc_next = dc_prev

        norm = np.sqrt(sum((gv ** 2).sum() for gv in g.values()))
        if norm > self.grad_clip:
            scale = self.grad_clip / (norm + 1e-12)
            g = {k: v * scale for k, v in g.items()}
        self.opt.step(self.p, g)
