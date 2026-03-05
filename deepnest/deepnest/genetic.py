import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .models import NestConfig


@dataclass
class _Individual:
    placement: List[Dict[str, Any]] = field(default_factory=list)
    rotation: List[float] = field(default_factory=list)
    fitness: Optional[float] = None
    processing: bool = False


class GeneticAlgorithm:
    def __init__(self, adam: List[Dict[str, Any]], config: NestConfig):
        self.config = config
        self.population: List[_Individual] = []

        # Respect the config.population_size more strictly.
        # Ensure we have at least some diversity (min 10), but don't explode
        # based on part count if the user/system asked for a specific limit.
        target_pop_size = max(10, config.population_size)

        # Only boost population based on part count if it doesn't exceed
        # a reasonable ceiling (e.g. 50) to prevent performance collapse.
        if len(adam) > 50:
            # Large part counts need efficient generations, not massive
            # populations
            pop_size = min(50, target_pop_size)
        else:
            # Smaller counts can afford slightly larger populations for
            # diversity
            pop_size = target_pop_size

        # First individual: use original orientation (all zeros)
        zero_angles = [0.0] * len(adam)
        self.population.append(
            _Individual(placement=adam.copy(), rotation=zero_angles)
        )

        # Second individual: use 90-degree rotations for some parts
        alt_angles = []
        for i in range(len(adam)):
            angle = 90.0 if i % 2 == 0 else 0.0
            alt_angles.append(angle)
        self.population.append(
            _Individual(placement=adam.copy(), rotation=alt_angles)
        )

        # Third individual: use 180-degree rotations
        rot180_angles = []
        for i in range(len(adam)):
            angle = 180.0 if i % 2 == 0 else 0.0
            rot180_angles.append(angle)
        self.population.append(
            _Individual(placement=adam.copy(), rotation=rot180_angles)
        )

        # Fourth individual: use random rotations for diversity
        angles = []
        for _ in adam:
            angle = random.randint(0, config.rotations - 1) * (
                360.0 / config.rotations
            )
            angles.append(angle)

        self.population.append(
            _Individual(placement=adam.copy(), rotation=angles)
        )

        while len(self.population) < pop_size:
            mutant = self._mutate(
                self.population[random.randint(0, len(self.population) - 1)]
            )
            self.population.append(mutant)

    def _mutate(self, individual: _Individual) -> _Individual:
        clone = _Individual(
            placement=individual.placement.copy(),
            rotation=individual.rotation.copy(),
        )

        for i in range(len(clone.placement)):
            rand = random.random()
            if rand < 0.01 * self.config.mutation_rate:
                j = i + 1
                if j < len(clone.placement):
                    clone.placement[i], clone.placement[j] = (
                        clone.placement[j],
                        clone.placement[i],
                    )

            rand = random.random()
            if rand < 0.01 * self.config.mutation_rate:
                clone.rotation[i] = random.randint(
                    0, self.config.rotations - 1
                ) * (360.0 / self.config.rotations)

        return clone

    def _mate(
        self, male: _Individual, female: _Individual
    ) -> Tuple[_Individual, _Individual]:
        min_len = min(len(male.placement), len(female.placement))
        if min_len == 0:
            return _Individual(), _Individual()

        cutpoint = int(
            max(1, min(min_len - 1, int(random.random() * min_len)))
        )

        gene1 = male.placement[:cutpoint]
        rot1 = male.rotation[:cutpoint]

        gene2 = female.placement[:cutpoint]
        rot2 = female.rotation[:cutpoint]

        male_ids = {p["id"] for p in gene1}
        female_ids = {p["id"] for p in gene2}

        for i, p in enumerate(female.placement):
            if p["id"] not in male_ids:
                gene1.append(p)
                rot1.append(female.rotation[i])

        for i, p in enumerate(male.placement):
            if p["id"] not in female_ids:
                gene2.append(p)
                rot2.append(male.rotation[i])

        return (
            _Individual(placement=gene1, rotation=rot1),
            _Individual(placement=gene2, rotation=rot2),
        )

    def generation(self) -> None:
        self.population.sort(
            key=lambda x: x.fitness if x.fitness is not None else float("inf")
        )

        new_population = [self.population[0]]

        while len(new_population) < len(self.population):
            male = self._random_weighted_individual()
            female = self._random_weighted_individual(exclude=male)

            child1, child2 = self._mate(male, female)

            new_population.append(self._mutate(child1))

            if len(new_population) < len(self.population):
                new_population.append(self._mutate(child2))

        self.population = new_population

    def _random_weighted_individual(
        self, exclude: Optional[_Individual] = None
    ) -> _Individual:
        pop = [p for p in self.population if p is not exclude]

        if not pop:
            return self.population[0]

        rand = random.random()

        lower = 0.0
        weight = 1.0 / len(pop)
        upper = weight

        for i, individual in enumerate(pop):
            if lower < rand < upper:
                return individual
            lower = upper
            upper += 2 * weight * ((len(pop) - i) / len(pop))

        return pop[0]
