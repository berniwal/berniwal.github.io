"""Genetic programming (mu+lambda evolution).

Each generation proposes a batch of offspring built from the current population by
tournament selection + subtree crossover/mutation; ``tell`` merges offspring into
the population and keeps the best (this truncation IS the elitism). Population
members are never re-evaluated, so every verifier call buys a *new* candidate --
keeping the budget comparison fair against the RL arms.

Lineage: tournament-selected evolution as a competitive alternative to RL on a
verifiable reward follows Real et al. 2019 (regularized/aging evolution for NAS).
An optional ``aging`` flag enables their age-based eviction.
"""
from __future__ import annotations

import copy

import numpy as np

from ..expression import (complexity, crossover_subtree, mutate_subtree,
                          random_tree)
from ..expression import Node
from ..verifier import Result
from .base import Proposer


class GeneticProgramming(Proposer):
    def __init__(self, task, rng, pop_size: int = 200, batch_size: int = 200,
                 tournament_size: int = 5, crossover_rate: float = 0.6,
                 mutation_rate: float = 0.3, immigrant_rate: float = 0.1,
                 max_depth: int = 4, max_complexity: int = 30,
                 aging: bool = False, **hp):
        super().__init__(task, rng, **hp)
        self.pop_size = pop_size
        self.batch_size = batch_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.immigrant_rate = immigrant_rate
        self.max_depth = max_depth
        self.max_complexity = max_complexity
        self.aging = aging
        self.pop: list[tuple[Node, float, int]] = []  # (tree, fitness, birth)
        self._gen = 0

    def _tournament(self) -> Node:
        idx = self.rng.integers(len(self.pop), size=self.tournament_size)
        winner = max(idx, key=lambda i: self.pop[i][1])
        return self.pop[winner][0]

    def _make_child(self) -> Node:
        if self.rng.random() < self.immigrant_rate:  # fresh blood vs. premature convergence
            return random_tree(self.task.grammar, self.rng, self.max_depth)
        r = self.rng.random()
        parent = self._tournament()
        if r < self.crossover_rate:
            child = crossover_subtree(parent, self._tournament(), self.rng)
        elif r < self.crossover_rate + self.mutation_rate:
            child = mutate_subtree(parent, self.task.grammar, self.rng, self.max_depth)
        else:
            child = copy.deepcopy(parent)  # reproduction
        if complexity(child) > self.max_complexity:  # bloat control
            child = random_tree(self.task.grammar, self.rng, max_depth=3)
        return child

    def ask(self) -> list[Node]:
        if not self.pop:  # generation 0: random initialization
            return [random_tree(self.task.grammar, self.rng, self.max_depth)
                    for _ in range(self.pop_size)]
        return [self._make_child() for _ in range(self.batch_size)]

    def tell(self, candidates: list[Node], results: list[Result]) -> None:
        self._gen += 1
        scored = [(c, r.reward, self._gen) for c, r in zip(candidates, results)]
        combined = self.pop + scored
        if self.aging:  # drop the oldest, then keep the fittest survivors
            combined.sort(key=lambda t: t[2], reverse=True)
            combined = combined[: max(self.pop_size * 2, len(scored))]
        combined.sort(key=lambda t: t[1], reverse=True)
        self.pop = combined[: self.pop_size]
