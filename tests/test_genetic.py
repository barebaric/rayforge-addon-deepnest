"""
Tests for rayforge.shared.deepnest.genetic module.
"""

import pytest

from rayforge.builtin_addons.deepnest.deepnest.deepnest.models import (
    NestConfig,
)
from rayforge.builtin_addons.deepnest.deepnest.deepnest.genetic import (
    GeneticAlgorithm,
)


def P(*points):
    """Helper to create a polygon from integer points."""
    return [(float(x), float(y)) for x, y in points]


class TestGeneticAlgorithm:
    @pytest.fixture
    def config(self):
        return NestConfig(population_size=5, rotations=4, mutation_rate=10)

    @pytest.fixture
    def sample_parts(self):
        return [
            {"id": 0, "source": 0, "polygon": P((0, 0), (10, 0), (5, 10))},
            {"id": 1, "source": 1, "polygon": P((0, 0), (8, 0), (4, 8))},
            {"id": 2, "source": 2, "polygon": P((0, 0), (6, 0), (3, 6))},
        ]

    def test_ga_initialization(self, config, sample_parts):
        ga = GeneticAlgorithm(sample_parts, config)
        assert len(ga.population) == config.population_size

    def test_ga_first_individual_has_correct_parts(self, config, sample_parts):
        ga = GeneticAlgorithm(sample_parts, config)
        first = ga.population[0]
        assert len(first.placement) == len(sample_parts)
        assert len(first.rotation) == len(sample_parts)

    def test_ga_rotations_in_valid_range(self, config, sample_parts):
        ga = GeneticAlgorithm(sample_parts, config)
        for ind in ga.population:
            for rot in ind.rotation:
                assert 0 <= rot < 360

    def test_ga_mutate(self, config, sample_parts):
        ga = GeneticAlgorithm(sample_parts, config)
        original = ga.population[0]
        mutant = ga._mutate(original)
        assert len(mutant.placement) == len(original.placement)
        assert len(mutant.rotation) == len(original.rotation)

    def test_ga_mate(self, config, sample_parts):
        ga = GeneticAlgorithm(sample_parts, config)
        male = ga.population[0]
        female = (
            ga.population[1] if len(ga.population) > 1 else ga.population[0]
        )

        child1, child2 = ga._mate(male, female)

        all_ids = {p["id"] for p in sample_parts}
        child1_ids = {p["id"] for p in child1.placement}
        child2_ids = {p["id"] for p in child2.placement}

        assert child1_ids == all_ids
        assert child2_ids == all_ids

    def test_ga_generation(self, config, sample_parts):
        ga = GeneticAlgorithm(sample_parts, config)

        for i, ind in enumerate(ga.population):
            ind.fitness = float(i + 1)

        ga.generation()

        assert len(ga.population) == config.population_size
