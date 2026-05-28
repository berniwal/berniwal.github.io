"""State buffer entry for the PUCT-style sampler (port of TTT-Discover's
`ttt_discover.tinker_utils.state.State`, slimmed for symbolic regression).

A State here is one *candidate solution* (an expression). PUCT picks which
state to refine next; the new rollouts are conditioned on that parent state
plus the task description.

This is candidate-refinement style PUCT (matches TTT-Discover's
`docs/intro.md`: "A state s is a candidate solution"), NOT prefix-of-reasoning
style. Each new rollout is a fresh prompt; no KV-cache sharing across rollouts.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class State:
    """One candidate in the PUCT buffer.

    Attributes
    ----------
    id : unique identifier (uuid4) — used for visit-count bookkeeping
    expr : the formula as an infix string (the "code" / "construction")
    value : the reward this state got from the verifier (None for unevaluated
            initial seeds)
    timestep : the training round when this state was first added
    parent_id : id of the parent State (None for initial seeds)
    parent_values : list of ancestor rewards, most-recent-first
                    (used for "before -> after" lines in the prompt)
    parents : full list of {"id": ..., "timestep": ...} ancestor refs,
              most-recent-first (used for lineage traversal in the sampler)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    expr: str = ""
    value: float | None = None
    timestep: int = 0
    parent_id: str | None = None
    parent_values: list[float] = field(default_factory=list)
    parents: list[dict] = field(default_factory=list)

    @classmethod
    def initial_seed(cls, timestep: int = 0) -> "State":
        """Create an initial-state seed (no parent, no value yet)."""
        return cls(timestep=timestep, expr="", value=None,
                   parent_id=None, parent_values=[], parents=[])

    def set_parent(self, parent: "State") -> None:
        """Mark `parent` as this state's immediate parent.

        Records parent_values (ancestor value chain, most-recent first) and
        parents (ancestor id refs, most-recent first) — both used by the
        sampler for visit-count propagation and lineage blocking.
        """
        if parent.value is not None:
            self.parent_values = [parent.value] + parent.parent_values
        self.parents = [{"id": parent.id, "timestep": parent.timestep}] + parent.parents
        self.parent_id = parent.id

    def to_dict(self) -> dict:
        return dict(id=self.id, expr=self.expr, value=self.value,
                    timestep=self.timestep, parent_id=self.parent_id,
                    parent_values=list(self.parent_values),
                    parents=list(self.parents))

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(id=d["id"], expr=d.get("expr", ""), value=d.get("value"),
                   timestep=d.get("timestep", 0),
                   parent_id=d.get("parent_id"),
                   parent_values=list(d.get("parent_values", [])),
                   parents=list(d.get("parents", [])))
