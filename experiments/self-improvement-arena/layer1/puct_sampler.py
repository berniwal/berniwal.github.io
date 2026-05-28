"""PUCT-style state sampler with lineage tracking.

Port of TTT-Discover's `ttt_discover.tinker_utils.sampler.PUCTSampler`, slimmed
for the single-task symbolic-regression setting. Verified line-by-line against
their `sampler.py` (see comments below).

The selection rule (matches their docstring):

    score(s) = Q(s) + c * scale * P(s) * sqrt(1 + T) / (1 + n(s))

where
    Q(s) = m[s] if n[s] > 0 else value(s)  -- optimistic max-of-children
    P(s) = rank-based prior over buffer values
    scale = max(R) - min(R) over non-initial states
    T = total expansions across the run
    n(s) = visit count, propagates to ancestors on each expansion

Diverges from AlphaZero PUCT in the four ways the paper footnote calls out:
(i) max-of-children Q, (ii) rank-based P, (iii) visit counts bubble to all
ancestors, (iv) lineage blocking inside a batch instead of virtual loss.

Tree logging: every call to `sample_states` and `update_states` appends one
JSON line to `tree_log_path` (if given) so the search can be visualised
post-hoc.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Optional

import numpy as np

from .puct_state import State


class PUCTSampler:
    def __init__(
        self,
        max_buffer_size: int = 200,
        seed_count: int = 4,          # initial-state copies in the buffer
        puct_c: float = 1.0,
        topk_children: int = 2,       # keep top-K children per parent in buffer
        tree_log_path: Optional[str] = None,
    ):
        self.max_buffer_size = int(max_buffer_size)
        self.seed_count = int(seed_count)
        self.puct_c = float(puct_c)
        self.topk_children = int(topk_children)
        self.tree_log_path = tree_log_path

        self._states: list[State] = []
        self._initial_states: list[State] = []
        self._lock = threading.Lock()

        # PUCT stats: id -> count / best-of-children
        self._n: dict[str, int] = {}
        self._m: dict[str, float] = {}
        self._T: int = 0           # total expansions (incl. failed)
        self._last_scale: float = 1.0
        # diagnostic: last sample's per-state breakdown
        self._last_picks: list[tuple[State, dict]] = []

        # Seed the buffer with `seed_count` copies of the initial state.
        # Each gets a fresh id so PUCT can pick them independently.
        for _ in range(self.seed_count):
            s = State.initial_seed(timestep=0)
            self._initial_states.append(s)
            self._states.append(s)

        if self.tree_log_path:
            os.makedirs(os.path.dirname(self.tree_log_path) or ".", exist_ok=True)
            # truncate any prior log
            open(self.tree_log_path, "w").close()

    # --- bookkeeping helpers -------------------------------------------------
    def _build_children_map(self) -> dict[str, set[str]]:
        children: dict[str, set[str]] = {}
        for s in self._states:
            for p in s.parents or []:
                pid = p.get("id")
                if pid:
                    children.setdefault(str(pid), set()).add(s.id)
        return children

    def _get_full_lineage(self, state: State, children_map: dict[str, set[str]]) -> set[str]:
        """state.id ∪ all ancestor ids ∪ all descendant ids of `state`.
        Note: does NOT include siblings of ancestors (matches TTT-Discover)."""
        lineage = {state.id}
        for p in state.parents or []:
            if p.get("id"):
                lineage.add(str(p["id"]))
        # BFS down from state through children_map
        queue = [state.id]
        visited = {state.id}
        while queue:
            sid = queue.pop(0)
            for cid in children_map.get(sid, []):
                if cid not in visited:
                    visited.add(cid)
                    lineage.add(cid)
                    queue.append(cid)
        return lineage

    def _compute_scale(self, vals: np.ndarray, mask: np.ndarray | None = None) -> float:
        v = vals[mask] if mask is not None else vals
        if v.size == 0:
            return 1.0
        return float(max(np.max(v) - np.min(v), 1e-6))

    def _compute_prior(self, vals: np.ndarray) -> np.ndarray:
        """Rank-based prior: top reward gets weight N, bottom gets 1, normalised."""
        if vals.size == 0:
            return np.array([])
        N = len(vals)
        ranks = np.argsort(np.argsort(-vals))      # 0 = best, N-1 = worst
        weights = (N - ranks).astype(np.float64)
        return weights / weights.sum()

    # --- public API ---------------------------------------------------------
    def sample_states(self, num_states: int, round_n: int | None = None) -> list[State]:
        """Pick `num_states` lineage-blocked states by PUCT score.

        Initial-state seeds with value=None are scored using -inf so they're only
        picked when the buffer is otherwise exhausted (after they get evaluated
        they get a real value and can be re-picked normally).
        """
        with self._lock:
            initial_ids = {s.id for s in self._initial_states}
            cands = list(self._states)
            if not cands:
                # should be unreachable given __init__ seeding, but safe-guard
                picked = [State.initial_seed(timestep=round_n or 0) for _ in range(num_states)]
                self._last_picks = [(s, {"n": 0, "Q": 0.0, "P": 0.0, "bonus": 0.0, "score": 0.0})
                                     for s in picked]
                return picked

            vals = np.array([float(s.value) if s.value is not None else float("-inf")
                              for s in cands])
            # finite values only (filter -inf for the scale calc)
            finite_mask = np.isfinite(vals)
            non_initial_mask = np.array([s.id not in initial_ids for s in cands])
            mask = finite_mask & non_initial_mask if (finite_mask & non_initial_mask).any() else finite_mask
            scale = self._compute_scale(vals, mask if mask.any() else None)
            self._last_scale = scale

            # Replace -inf with 0 in vals just for the rank computation;
            # initial unevaluated states will end up with the worst rank.
            vals_for_rank = np.where(finite_mask, vals, np.min(vals[finite_mask]) - 1.0
                                                       if finite_mask.any() else 0.0)
            P = self._compute_prior(vals_for_rank)

            sqrtT = float(np.sqrt(1.0 + self._T))

            scored: list[tuple[float, float, State, dict]] = []
            for i, s in enumerate(cands):
                n = self._n.get(s.id, 0)
                m = self._m.get(s.id, float(vals[i]))
                Q = m if n > 0 else float(vals[i])
                bonus = self.puct_c * scale * float(P[i]) * sqrtT / (1.0 + n)
                score = Q + bonus
                scored.append((score, float(vals[i]), s,
                                dict(n=n, Q=Q, P=float(P[i]), bonus=bonus, score=score)))

            # Sort by score desc, value desc as tiebreak
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

            picked: list[State] = []
            picks_meta: list[tuple[State, dict]] = []
            if num_states <= 1:
                top = scored[:num_states]
                picked = [t[2] for t in top]
                picks_meta = [(t[2], t[3]) for t in top]
            else:
                children_map = self._build_children_map()
                blocked: set[str] = set()
                for sc, _, s, meta in scored:
                    if s.id in blocked:
                        continue
                    picked.append(s)
                    picks_meta.append((s, meta))
                    blocked.update(self._get_full_lineage(s, children_map))
                    if len(picked) >= num_states:
                        break

            self._last_picks = picks_meta
            self._log_event(round_n, "sample", picks_meta)
            return picked

    def update_states(self, new_children: list[State], parent_states: list[State],
                       round_n: int | None = None, save: bool = True) -> None:
        """Add new evaluated children to the buffer.

        - Bumps n(s) for the parent AND all its ancestors (visit count
          propagates up the tree).
        - Tracks m[parent] = max over child values (max-of-children Q).
        - Increments T (total expansions) once per (parent_id, batch).
        - Dedups on expr.
        - Optionally enforces topk_children per parent: only the top-K children
          (by value) of each parent are retained in the buffer.
        - Optionally trims buffer to max_buffer_size by reward.
        """
        assert len(new_children) == len(parent_states)
        if not new_children:
            return

        with self._lock:
            parent_max: dict[str, float] = {}
            parent_obj: dict[str, State] = {}
            for child, parent in zip(new_children, parent_states):
                if child.value is None:
                    continue
                pid = parent.id
                parent_obj[pid] = parent
                parent_max[pid] = max(parent_max.get(pid, float("-inf")), float(child.value))

            for pid, y in parent_max.items():
                self._m[pid] = max(self._m.get(pid, y), y)
                parent = parent_obj[pid]
                anc_ids = [pid] + [str(p["id"]) for p in (parent.parents or []) if p.get("id")]
                for aid in anc_ids:
                    self._n[aid] = self._n.get(aid, 0) + 1
                self._T += 1

            # set parent info on each child and dedup by expr
            existing_exprs = {s.expr for s in self._states if s.expr}
            additions: list[tuple[State, State]] = []
            for child, parent in zip(new_children, parent_states):
                if child.value is None or not child.expr:
                    continue
                child.set_parent(parent)
                if child.expr in existing_exprs:
                    continue
                existing_exprs.add(child.expr)
                additions.append((child, parent))

            # topk children per parent
            if self.topk_children > 0 and additions:
                by_parent: dict[str, list[tuple[State, State]]] = {}
                for c, p in additions:
                    by_parent.setdefault(p.id, []).append((c, p))
                kept: list[tuple[State, State]] = []
                for cs in by_parent.values():
                    cs.sort(key=lambda x: x[0].value if x[0].value is not None
                                                       else float("-inf"), reverse=True)
                    kept.extend(cs[:self.topk_children])
                additions = kept

            for c, _ in additions:
                self._states.append(c)

            if len(self._states) > self.max_buffer_size:
                # Trim to max_buffer_size by reward, ALWAYS keeping initial seeds.
                initial_ids = {s.id for s in self._initial_states}
                non_initial = [(s, s.value if s.value is not None else float("-inf"))
                               for s in self._states if s.id not in initial_ids]
                non_initial.sort(key=lambda x: x[1], reverse=True)
                budget = self.max_buffer_size - len(self._initial_states)
                kept_non_initial = [s for s, _ in non_initial[:max(budget, 0)]]
                self._states = list(self._initial_states) + kept_non_initial

            self._log_event(round_n, "update", [(c, {"value": c.value}) for c, _ in additions])

    def record_failed_rollout(self, parent: State) -> None:
        """A rollout produced no parseable candidate: still counts as a visit
        for `parent` and all its ancestors (matches their convention)."""
        with self._lock:
            anc_ids = [parent.id] + [str(p["id"]) for p in (parent.parents or []) if p.get("id")]
            for aid in anc_ids:
                self._n[aid] = self._n.get(aid, 0) + 1
            self._T += 1

    # --- introspection ------------------------------------------------------
    def buffer_size(self) -> int:
        return len(self._states)

    def total_expansions(self) -> int:
        return self._T

    def get_sample_stats(self) -> dict:
        vals = np.array([s.value for s in self._states if s.value is not None])
        if vals.size == 0:
            return dict(buffer_size=len(self._states), T=self._T, scale=self._last_scale)
        return dict(buffer_size=len(self._states), T=self._T, scale=self._last_scale,
                    value_mean=float(vals.mean()), value_max=float(vals.max()),
                    value_min=float(vals.min()))

    # --- tree logging -------------------------------------------------------
    def _log_event(self, round_n: int | None, kind: str,
                    entries: list[tuple[State, dict]]) -> None:
        if not self.tree_log_path or round_n is None:
            return
        rec = dict(round=round_n, kind=kind, T=self._T,
                    buffer_size=len(self._states),
                    entries=[dict(id=s.id, parent_id=s.parent_id,
                                   expr=s.expr, value=s.value,
                                   timestep=s.timestep, **meta)
                              for s, meta in entries])
        with open(self.tree_log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
