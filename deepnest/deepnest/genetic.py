from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .models import NestConfig

from raygeo.geo.algo.nest2d.genetic import GeneticAlgorithm as _RustGA


@dataclass
class _Individual:
    placement: List[Dict[str, Any]] = field(default_factory=list)
    rotation: List[float] = field(default_factory=list)
    flip_h: List[bool] = field(default_factory=list)
    flip_v: List[bool] = field(default_factory=list)
    fitness: Optional[float] = None
    processing: bool = False


class GeneticAlgorithm:
    def __init__(self, adam: List[Dict[str, Any]], config: NestConfig):
        self.config = config

        # Derive population size (matches original logic)
        target_pop_size = max(10, config.population_size)
        if len(adam) > 50:
            pop_size = min(50, target_pop_size)
        else:
            pop_size = target_pop_size

        num_parts = len(adam)
        self._ga = _RustGA(
            num_parts,
            {
                "rotation_count": config.rotations,
                "flip_h": bool(config.flip_h),
                "flip_v": bool(config.flip_v),
                "population_size": int(pop_size),
                "mutation_rate": float(config.mutation_rate),
            },
        )

        self.population: List[_Individual] = []
        for i in range(len(self._ga)):
            rotations, flips_h, flips_v, fitness = self._ga.get_individual(i)
            self.population.append(
                _Individual(
                    placement=adam.copy(),
                    rotation=rotations,
                    flip_h=flips_h,
                    flip_v=flips_v,
                    fitness=None if fitness == float("inf") else fitness,
                )
            )

    def _sync_to_rust(self) -> None:
        """Push Python-side fitness values back to Rust before generation()."""
        for i, ind in enumerate(self.population):
            fitness = ind.fitness if ind.fitness is not None else float("inf")
            self._ga.set_fitness(i, fitness)

    def _sync_from_rust(self) -> None:
        """Pull Rust-side genomes back to Python after generation()."""
        for i, ind in enumerate(self.population):
            rotations, flips_h, flips_v, fitness = self._ga.get_individual(i)
            ind.rotation = rotations
            ind.flip_h = flips_h
            ind.flip_v = flips_v
            ind.fitness = None if fitness == float("inf") else fitness

    def generation(self) -> None:
        self._sync_to_rust()
        self._ga.generation()
        self._sync_from_rust()
