"""Proposer registry: name -> class. The runner looks methods up here by name."""
from __future__ import annotations

from .base import Proposer
from .genetic import GeneticProgramming
from .random_search import RandomSearch
from .rl_greedy import RLGreedy
from .rl_risk import RLRisk

REGISTRY: dict[str, type[Proposer]] = {
    "random": RandomSearch,
    "gp": GeneticProgramming,
    "greedy": RLGreedy,
    "risk": RLRisk,
}


def get_proposer(name: str) -> type[Proposer]:
    if name not in REGISTRY:
        raise ValueError(f"unknown proposer {name!r}; choices: {list(REGISTRY)}")
    return REGISTRY[name]
