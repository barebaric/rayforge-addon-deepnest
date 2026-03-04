"""
Tests for rayforge.shared.deepnest.models module.
"""

from deepnest.deepnest.models import NestConfig


class TestNestConfig:
    def test_default_config(self):
        config = NestConfig()
        assert config.curve_tolerance == 0.05
        assert config.spacing == 0.0
        assert config.rotations == 36
        assert config.population_size == 10
        assert config.mutation_rate == 10
        assert config.placement_type == "gravity"

    def test_custom_config(self):
        config = NestConfig(
            curve_tolerance=0.5,
            spacing=1.0,
            rotations=8,
            population_size=20,
        )
        assert config.curve_tolerance == 0.5
        assert config.spacing == 1.0
        assert config.rotations == 8
        assert config.population_size == 20

    def test_config_immutability(self):
        config = NestConfig()
        original_tolerance = config.curve_tolerance
        assert config.curve_tolerance == original_tolerance
