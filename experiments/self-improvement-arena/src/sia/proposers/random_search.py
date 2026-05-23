"""Random search baseline: sample fresh random trees every batch, never learn."""
from __future__ import annotations

from ..expression import Node, random_tree
from ..verifier import Result
from .base import Proposer


class RandomSearch(Proposer):
    def __init__(self, task, rng, batch_size: int = 200, max_depth: int = 4, **hp):
        super().__init__(task, rng, **hp)
        self.batch_size = batch_size
        self.max_depth = max_depth

    def ask(self) -> list[Node]:
        return [random_tree(self.task.grammar, self.rng, self.max_depth)
                for _ in range(self.batch_size)]

    def tell(self, candidates: list[Node], results: list[Result]) -> None:
        pass
