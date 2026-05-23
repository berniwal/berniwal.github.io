"""The Proposer interface -- the ONLY thing that changes between methods and
between Layer 0 (search/RL) and Layer 1 (LLM). The runner drives an ask/tell loop:

    candidates = proposer.ask()          # propose a batch
    results = [verifier(c) for c in ...] # scored by the shared verifier
    proposer.tell(candidates, results)   # learn from the rewards

This keeps the comparison fair: every method is just a different ask/tell.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..expression import Node
from ..task import Task
from ..verifier import Result


class Proposer(ABC):
    def __init__(self, task: Task, rng: np.random.Generator, **hp):
        self.task = task
        self.rng = rng
        self.hp = hp

    @abstractmethod
    def ask(self) -> list[Node]:
        """Return a batch of candidate expressions to be evaluated."""

    @abstractmethod
    def tell(self, candidates: list[Node], results: list[Result]) -> None:
        """Update internal state from the scored candidates (noop for random)."""
