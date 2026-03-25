import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .models import NestConfig


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
        no_flip = [False] * len(adam)
        self.population.append(
            _Individual(
                placement=adam.copy(),
                rotation=zero_angles,
                flip_h=no_flip.copy(),
                flip_v=no_flip.copy(),
            )
        )

        # Second individual: use alternate rotations if allowed
        alt_angles = []
        if config.rotations >= 4:
            alt_step = 360.0 / config.rotations
            for i in range(len(adam)):
                angle = alt_step if i % 2 == 0 else 0.0
                alt_angles.append(angle)
        else:
            alt_angles = [0.0] * len(adam)
        self.population.append(
            _Individual(
                placement=adam.copy(),
                rotation=alt_angles,
                flip_h=no_flip.copy(),
                flip_v=no_flip.copy(),
            )
        )

        # Third individual: use another rotation angle if allowed
        rot180_angles = []
        if config.rotations >= 2:
            half_step = (360.0 / config.rotations) * (config.rotations // 2)
            for i in range(len(adam)):
                angle = half_step if i % 2 == 0 else 0.0
                rot180_angles.append(angle)
        else:
            rot180_angles = [0.0] * len(adam)
        self.population.append(
            _Individual(
                placement=adam.copy(),
                rotation=rot180_angles,
                flip_h=no_flip.copy(),
                flip_v=no_flip.copy(),
            )
        )

        # Fourth individual: use random rotations and flips for diversity
        angles = []
        flips_h = []
        flips_v = []
        for _ in adam:
            angle = random.randint(0, config.rotations - 1) * (
                360.0 / config.rotations
            )
            angles.append(angle)
            flips_h.append(config.flip_h and random.choice([True, False]))
            flips_v.append(config.flip_v and random.choice([True, False]))

        self.population.append(
            _Individual(
                placement=adam.copy(),
                rotation=angles,
                flip_h=flips_h,
                flip_v=flips_v,
            )
        )

        # Fifth individual: try horizontal flip on all parts if allowed
        if config.flip_h:
            self.population.append(
                _Individual(
                    placement=adam.copy(),
                    rotation=zero_angles.copy(),
                    flip_h=[True] * len(adam),
                    flip_v=no_flip.copy(),
                )
            )

        # Sixth individual: try vertical flip on all parts if allowed
        if config.flip_v:
            self.population.append(
                _Individual(
                    placement=adam.copy(),
                    rotation=zero_angles.copy(),
                    flip_h=no_flip.copy(),
                    flip_v=[True] * len(adam),
                )
            )

        # Seventh individual: try both flips on all parts if allowed
        if config.flip_h and config.flip_v:
            self.population.append(
                _Individual(
                    placement=adam.copy(),
                    rotation=zero_angles.copy(),
                    flip_h=[True] * len(adam),
                    flip_v=[True] * len(adam),
                )
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
            flip_h=individual.flip_h.copy(),
            flip_v=individual.flip_v.copy(),
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

            if self.config.flip_h:
                rand = random.random()
                if rand < 0.05:
                    clone.flip_h[i] = not clone.flip_h[i]

            if self.config.flip_v:
                rand = random.random()
                if rand < 0.05:
                    clone.flip_v[i] = not clone.flip_v[i]

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
        flip_h1 = male.flip_h[:cutpoint]
        flip_v1 = male.flip_v[:cutpoint]

        gene2 = female.placement[:cutpoint]
        rot2 = female.rotation[:cutpoint]
        flip_h2 = female.flip_h[:cutpoint]
        flip_v2 = female.flip_v[:cutpoint]

        male_ids = {p["id"] for p in gene1}
        female_ids = {p["id"] for p in gene2}

        for i, p in enumerate(female.placement):
            if p["id"] not in male_ids:
                gene1.append(p)
                rot1.append(female.rotation[i])
                flip_h1.append(female.flip_h[i])
                flip_v1.append(female.flip_v[i])

        for i, p in enumerate(male.placement):
            if p["id"] not in female_ids:
                gene2.append(p)
                rot2.append(male.rotation[i])
                flip_h2.append(male.flip_h[i])
                flip_v2.append(male.flip_v[i])

        return (
            _Individual(
                placement=gene1, rotation=rot1, flip_h=flip_h1, flip_v=flip_v1
            ),
            _Individual(
                placement=gene2, rotation=rot2, flip_h=flip_h2, flip_v=flip_v2
            ),
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
