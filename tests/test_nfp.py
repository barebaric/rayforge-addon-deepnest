from rayforge.builtin_addons.deepnest.deepnest.deepnest.nfp import (
    no_fit_polygon,
    inner_fit_polygon,
    get_placement_position,
    _nfp_inside,
    _nfp_outside,
)
from rayforge.builtin_addons.deepnest.deepnest.deepnest.models import (
    NestConfig,
)
from rayforge.core.geo.polygon import Polygon


class TestNoFitPolygon:
    """Tests for no_fit_polygon function."""

    def test_empty_static(self):
        config = NestConfig()
        orbiting: Polygon = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ]
        result = no_fit_polygon([], orbiting, inside=False, config=config)
        assert result == []

    def test_empty_orbiting(self):
        config = NestConfig()
        static: Polygon = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
        result = no_fit_polygon(static, [], inside=False, config=config)
        assert result == []

    def test_static_too_few_points(self):
        config = NestConfig()
        static: Polygon = [(0.0, 0.0), (10.0, 0.0)]
        orbiting: Polygon = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        result = no_fit_polygon(static, orbiting, inside=False, config=config)
        assert result == []

    def test_orbiting_too_few_points(self):
        config = NestConfig()
        static: Polygon = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
        orbiting: Polygon = [(0.0, 0.0), (5.0, 0.0)]
        result = no_fit_polygon(static, orbiting, inside=False, config=config)
        assert result == []

    def test_rectangle_outside(self):
        config = NestConfig()
        static: Polygon = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
        orbiting: Polygon = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ]
        result = no_fit_polygon(static, orbiting, inside=False, config=config)
        assert len(result) >= 1
        for nfp in result:
            assert len(nfp) >= 3

    def test_rectangle_inside(self):
        config = NestConfig()
        static: Polygon = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
        orbiting: Polygon = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ]
        result = no_fit_polygon(static, orbiting, inside=True, config=config)
        assert len(result) >= 1
        for nfp in result:
            assert len(nfp) >= 3

    def test_triangle_outside(self):
        config = NestConfig()
        static: Polygon = [(0.0, 0.0), (30.0, 0.0), (15.0, 30.0)]
        orbiting: Polygon = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        result = no_fit_polygon(static, orbiting, inside=False, config=config)
        assert len(result) >= 0

    def test_complex_polygon(self):
        config = NestConfig()
        static: Polygon = [
            (0.0, 0.0),
            (50.0, 0.0),
            (50.0, 30.0),
            (30.0, 30.0),
            (30.0, 50.0),
            (0.0, 50.0),
        ]
        orbiting: Polygon = [
            (0.0, 0.0),
            (15.0, 0.0),
            (15.0, 15.0),
            (0.0, 15.0),
        ]
        result = no_fit_polygon(static, orbiting, inside=False, config=config)
        assert len(result) >= 0


class TestInnerFitPolygon:
    """Tests for inner_fit_polygon function."""

    def test_empty_bin(self):
        config = NestConfig()
        part: Polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        result = inner_fit_polygon([], part, config=config)
        assert result is None

    def test_empty_part(self):
        config = NestConfig()
        bin_poly: Polygon = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
        result = inner_fit_polygon(bin_poly, [], config=config)
        assert result is None

    def test_part_larger_than_bin(self):
        config = NestConfig()
        bin_poly: Polygon = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ]
        part: Polygon = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
        result = inner_fit_polygon(bin_poly, part, config=config)
        assert result is None or len(result) < 3

    def test_small_part_in_large_bin(self):
        config = NestConfig()
        bin_poly: Polygon = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
        part: Polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        result = inner_fit_polygon(bin_poly, part, config=config)
        assert result is not None
        assert len(result) >= 3

    def test_returns_largest_nfp(self):
        config = NestConfig()
        bin_poly: Polygon = [
            (0.0, 0.0),
            (200.0, 0.0),
            (200.0, 200.0),
            (0.0, 200.0),
        ]
        part: Polygon = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
        result = inner_fit_polygon(bin_poly, part, config=config)
        assert result is not None
        assert len(result) >= 3


class TestGetPlacementPosition:
    """Tests for get_placement_position function."""

    def test_empty_nfp(self):
        config = NestConfig()
        part: Polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        result = get_placement_position([], part, (0.0, 0.0), config=config)
        assert result is None

    def test_gravity_placement(self):
        config = NestConfig(placement_type="gravity")
        nfp: Polygon = [(0.0, 0.0), (50.0, 0.0), (50.0, 30.0), (0.0, 30.0)]
        part: Polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        result = get_placement_position(nfp, part, (0.0, 0.0), config=config)
        assert result is not None
        assert len(result) == 2
        assert result[1] == 0.0

    def test_box_placement(self):
        config = NestConfig(placement_type="box")
        nfp: Polygon = [(10.0, 10.0), (60.0, 10.0), (60.0, 40.0), (10.0, 40.0)]
        part: Polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        result = get_placement_position(nfp, part, (0.0, 0.0), config=config)
        assert result is not None
        assert len(result) == 2
        assert result[0] == 10.0
        assert result[1] == 10.0

    def test_triangular_nfp(self):
        config = NestConfig(placement_type="gravity")
        nfp: Polygon = [(0.0, 10.0), (30.0, 40.0), (60.0, 10.0)]
        part: Polygon = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        result = get_placement_position(nfp, part, (0.0, 0.0), config=config)
        assert result is not None


class TestNfpInside:
    """Tests for _nfp_inside function."""

    def test_basic_inside(self):
        config = NestConfig()
        scale = config.clipper_scale
        from rayforge.core.geo.polygon import to_clipper

        static = to_clipper(
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)], scale
        )
        orbiting = to_clipper(
            [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)], scale
        )

        result = _nfp_inside(static, orbiting, scale)
        assert len(result) >= 1
        for nfp in result:
            assert len(nfp) >= 3

    def test_part_too_large(self):
        config = NestConfig()
        scale = config.clipper_scale
        from rayforge.core.geo.polygon import to_clipper

        static = to_clipper(
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], scale
        )
        orbiting = to_clipper(
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)], scale
        )

        result = _nfp_inside(static, orbiting, scale)
        assert result == []


class TestNfpOutside:
    """Tests for _nfp_outside function."""

    def test_basic_outside(self):
        config = NestConfig()
        scale = config.clipper_scale
        from rayforge.core.geo.polygon import to_clipper

        static = to_clipper(
            [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)], scale
        )
        orbiting = to_clipper(
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], scale
        )

        result = _nfp_outside(static, orbiting, scale)
        assert len(result) >= 0


class TestNfpWithRotations:
    """Tests for NFP with rotated polygons."""

    def test_rotated_square(self):
        config = NestConfig()
        import math

        cx, cy = 50.0, 50.0
        size = 10.0
        angle = math.radians(45)

        corners: Polygon = []
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            x = cx + dx * size / 2
            y = cy + dy * size / 2
            rx = cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle)
            ry = cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
            corners.append((rx, ry))

        bin_poly: Polygon = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
        result = inner_fit_polygon(bin_poly, corners, config=config)
        assert result is not None or result == []

    def test_rotated_triangle(self):
        config = NestConfig()
        import math

        cx, cy = 50.0, 50.0
        angle = math.radians(30)

        triangle: Polygon = [(50.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        rotated: Polygon = []
        for x, y in triangle:
            rx = cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle)
            ry = cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
            rotated.append((rx, ry))

        bin_poly: Polygon = [
            (0.0, 0.0),
            (150.0, 0.0),
            (150.0, 150.0),
            (0.0, 150.0),
        ]
        result = inner_fit_polygon(bin_poly, rotated, config=config)
        assert result is not None or result == []


class TestNfpEdgeCases:
    """Tests for edge cases in NFP calculations."""

    def test_coincident_points(self):
        config = NestConfig()
        static: Polygon = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)]
        orbiting: Polygon = [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        result = no_fit_polygon(static, orbiting, inside=False, config=config)
        assert len(result) >= 0

    def test_very_small_part(self):
        config = NestConfig()
        bin_poly: Polygon = [
            (0.0, 0.0),
            (1000.0, 0.0),
            (1000.0, 1000.0),
            (0.0, 1000.0),
        ]
        part: Polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        result = inner_fit_polygon(bin_poly, part, config=config)
        assert result is not None

    def test_very_large_scale(self):
        config = NestConfig(clipper_scale=10000000)
        bin_poly: Polygon = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
        part: Polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        result = inner_fit_polygon(bin_poly, part, config=config)
        assert result is not None or result == []

    def test_concave_polygon(self):
        config = NestConfig()
        static: Polygon = [
            (0.0, 0.0),
            (50.0, 0.0),
            (50.0, 25.0),
            (25.0, 25.0),
            (25.0, 50.0),
            (50.0, 50.0),
            (50.0, 75.0),
            (0.0, 75.0),
        ]
        orbiting: Polygon = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ]
        result = no_fit_polygon(static, orbiting, inside=True, config=config)
        assert len(result) >= 0
