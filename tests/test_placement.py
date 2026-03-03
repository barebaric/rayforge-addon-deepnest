"""
Tests for rayforge.shared.deepnest.placement module.
"""

from __future__ import annotations

from typing import Union, List, cast

import pytest
import pyclipper

from rayforge.builtin_addons.deepnest.deepnest.deepnest.models import (
    NestConfig,
    Placement,
    SheetInfo,
)
from rayforge.builtin_addons.deepnest.deepnest.deepnest.placement import (
    NestResult,
    place_parts,
)
from rayforge.core.geo.polygon import to_clipper, Polygon


def P(*points):
    """Helper to create a polygon from integer points."""
    return [(float(x), float(y)) for x, y in points]


def SI(polygon: Polygon, uid: str = "sheet") -> SheetInfo:
    """Helper to create a SheetInfo from a polygon."""
    return SheetInfo(uid=uid, polygon=polygon)


def make_sheets(*polygons: Polygon) -> List[SheetInfo]:
    """Helper to create a list of SheetInfo from polygons."""
    return [
        SheetInfo(uid=f"sheet-{i}", polygon=p) for i, p in enumerate(polygons)
    ]


def as_sheets(
    sheets_like: Union[SheetInfo, List[SheetInfo], List[Polygon], Polygon],
) -> List[SheetInfo]:
    """Convert a polygon or list of polygons to SheetInfo list."""
    if isinstance(sheets_like, SheetInfo):
        return [sheets_like]
    if isinstance(sheets_like, list):
        if len(sheets_like) == 0:
            return []
        first = sheets_like[0]
        if isinstance(first, SheetInfo):
            return cast(List[SheetInfo], sheets_like)
        if isinstance(first, tuple):
            single_poly = cast(Polygon, sheets_like)
            return [SheetInfo(uid="sheet-0", polygon=single_poly)]
        polygons = cast(List[Polygon], sheets_like)
        return [
            SheetInfo(uid=f"sheet-{i}", polygon=s)
            for i, s in enumerate(polygons)
        ]
    raise TypeError(f"Unexpected type: {type(sheets_like)}")


def polygons_intersect(poly1, poly2, min_overlap=0.1):
    """
    Check if two polygons have significant intersection.
    Returns the intersection area, or 0 if no intersection.
    """
    scale = 10000000
    try:
        c1 = to_clipper(poly1, scale)
        c2 = to_clipper(poly2, scale)

        clipper = pyclipper.Pyclipper()
        clipper.AddPath(c1, pyclipper.PT_SUBJECT, True)
        clipper.AddPath(c2, pyclipper.PT_CLIP, True)

        result = clipper.Execute(
            pyclipper.CT_INTERSECTION,
            pyclipper.PFT_NONZERO,
            pyclipper.PFT_NONZERO,
        )

        if result:
            total_area = sum(abs(pyclipper.Area(r)) for r in result)
            area = total_area / (scale * scale)
            return area if area > min_overlap else 0
    except Exception:
        pass
    return 0


class TestPlacement:
    def test_default(self):
        p = Placement(
            id=0,
            source=0,
            uid="test-uid",
            x=1.0,
            y=2.0,
            rotation=90.0,
            polygons=[P((0, 0), (10, 0), (5, 10))],
        )
        assert p.id == 0
        assert p.source == 0
        assert p.uid == "test-uid"
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.rotation == 90.0
        assert len(p.polygons) == 1


class TestNestResult:
    def test_default(self):
        result = NestResult(
            placements=[], fitness=1.0, area_used=100.0, sheet_index=0
        )
        assert result.placements == []
        assert result.fitness == 1.0
        assert result.area_used == 100.0
        assert result.sheet_index == 0


class TestPlacePartsBasic:
    @pytest.fixture
    def config(self):
        return NestConfig(placement_type="gravity", curve_tolerance=0.5)

    @pytest.fixture
    def sample_parts(self):
        return [
            {
                "id": 0,
                "source": 0,
                "uid": "part-0",
                "polygons": [P((0, 0), (10, 0), (10, 10), (0, 10))],
            },
            {
                "id": 1,
                "source": 1,
                "uid": "part-1",
                "polygons": [P((0, 0), (8, 0), (8, 8), (0, 8))],
            },
        ]

    @pytest.fixture
    def sample_sheet(self):
        return SI(P((0, 0), (100, 0), (100, 100), (0, 100)))

    @pytest.fixture
    def sample_sheets(self, request):
        return make_sheets(P((0, 0), (100, 0), (100, 100), (0, 100)))

    def test_place_parts_basic(self, config, sample_parts, sample_sheet):
        rotations = [0.0, 0.0]
        result = place_parts(sample_parts, [sample_sheet], rotations, config)

        assert result is not None
        assert len(result.placements) >= 1

    def test_place_parts_with_rotation(
        self, config, sample_parts, sample_sheet
    ):
        rotations = [90.0, 0.0]
        result = place_parts(sample_parts, [sample_sheet], rotations, config)

        assert result is not None

    def test_place_parts_empty_parts(self, config, sample_sheet):
        result = place_parts([], [sample_sheet], [], config)
        assert result is None

    def test_place_parts_empty_sheets(self, config, sample_parts):
        result = place_parts(sample_parts, [], [0.0, 0.0], config)
        assert result is None

    def test_fitness_calculation(self, config, sample_parts, sample_sheet):
        rotations = [0.0, 0.0]
        result = place_parts(sample_parts, [sample_sheet], rotations, config)

        assert result is not None
        assert result.fitness > 0

    def test_placements_have_correct_ids(
        self, config, sample_parts, sample_sheet
    ):
        rotations = [0.0, 0.0]
        result = place_parts(sample_parts, [sample_sheet], rotations, config)

        assert result is not None
        placed_ids = {p.id for p in result.placements}
        expected_ids = {p["id"] for p in sample_parts}
        assert placed_ids == expected_ids

    def test_multi_polygon_part(self, config, sample_sheet):
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "multi-part",
                "polygons": [
                    P((0, 0), (10, 0), (10, 10), (0, 10)),
                    P((20, 0), (30, 0), (30, 10), (20, 10)),
                ],
            },
        ]
        rotations = [0.0]
        result = place_parts(parts, [sample_sheet], rotations, config)

        assert result is not None
        assert len(result.placements) == 1
        assert len(result.placements[0].polygons) == 2


class TestNoOverlap:
    """Test that placed parts don't overlap - critical for real-world use."""

    @pytest.fixture
    def config(self):
        return NestConfig(placement_type="gravity", curve_tolerance=0.5)

    def test_two_identical_squares_no_overlap(self, config):
        """Two identical squares should not overlap."""
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "sq-0",
                "polygons": [P((0, 0), (10, 0), (10, 10), (0, 10))],
            },
            {
                "id": 1,
                "source": 1,
                "uid": "sq-1",
                "polygons": [P((0, 0), (10, 0), (10, 10), (0, 10))],
            },
        ]
        sheet = P((0, 0), (100, 0), (100, 100), (0, 100))
        rotations = [0.0, 0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) == 2

        for poly1 in result.placements[0].polygons:
            for poly2 in result.placements[1].polygons:
                overlap = polygons_intersect(poly1, poly2)
                assert overlap == 0, f"Parts overlap by {overlap} square units"

    def test_ten_identical_parts_no_overlap(self, config):
        """Ten identical rectangles should not overlap."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"rect-{i}",
                "polygons": [P((0, 0), (10, 0), (10, 5), (0, 5))],
            }
            for i in range(10)
        ]
        sheet = P((0, 0), (100, 0), (100, 100), (0, 100))
        rotations = [0.0] * 10

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) == 10

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Parts {p1.uid} and {p2.uid} overlap by {overlap}"
                        )

    def test_different_sizes_no_overlap(self, config):
        """Parts of different sizes should not overlap."""
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "large",
                "polygons": [P((0, 0), (20, 0), (20, 20), (0, 20))],
            },
            {
                "id": 1,
                "source": 1,
                "uid": "medium",
                "polygons": [P((0, 0), (10, 0), (10, 10), (0, 10))],
            },
            {
                "id": 2,
                "source": 2,
                "uid": "small",
                "polygons": [P((0, 0), (5, 0), (5, 5), (0, 5))],
            },
        ]
        sheet = P((0, 0), (100, 0), (100, 100), (0, 100))
        rotations = [0.0, 0.0, 0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) == 3

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Parts {p1.uid} and {p2.uid} overlap by {overlap}"
                        )

    def test_rotated_parts_no_overlap(self, config):
        """Rotated parts should not overlap."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"rect-{i}",
                "polygons": [P((0, 0), (15, 0), (15, 5), (0, 5))],
            }
            for i in range(6)
        ]
        sheet = P((0, 0), (100, 0), (100, 100), (0, 100))
        rotations = [0.0, 90.0, 180.0, 270.0, 0.0, 90.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Parts {p1.uid} (rot={p1.rotation}) and "
                            f"{p2.uid} (rot={p2.rotation}) "
                            f"overlap by {overlap}"
                        )


class TestRealWorldScenarios:
    """Test cases that mirror real-world nesting scenarios."""

    @pytest.fixture
    def config(self):
        return NestConfig(
            placement_type="gravity",
            curve_tolerance=0.5,
            rotations=4,
            spacing=0.5,
        )

    def test_laser_cut_parts(self, config):
        """Simulate laser-cut parts with typical shapes."""
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "bracket",
                "polygons": [
                    P(
                        (0, 0),
                        (50, 0),
                        (50, 10),
                        (40, 10),
                        (40, 30),
                        (50, 30),
                        (50, 40),
                        (0, 40),
                        (0, 30),
                        (10, 30),
                        (10, 10),
                        (0, 10),
                    )
                ],
            },
            {
                "id": 1,
                "source": 1,
                "uid": "bracket-2",
                "polygons": [
                    P(
                        (0, 0),
                        (50, 0),
                        (50, 10),
                        (40, 10),
                        (40, 30),
                        (50, 30),
                        (50, 40),
                        (0, 40),
                        (0, 30),
                        (10, 30),
                        (10, 10),
                        (0, 10),
                    )
                ],
            },
            {
                "id": 2,
                "source": 2,
                "uid": "washer",
                "polygons": [P((0, 0), (20, 0), (20, 20), (0, 20))],
            },
            {
                "id": 3,
                "source": 3,
                "uid": "washer-2",
                "polygons": [P((0, 0), (20, 0), (20, 20), (0, 20))],
            },
        ]
        sheet = P((0, 0), (200, 0), (200, 200), (0, 200))
        rotations = [0.0, 90.0, 0.0, 0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(
                            poly1, poly2, min_overlap=0.5
                        )
                        assert overlap == 0


class TestPositionUniqueness:
    """Test that parts are placed at unique positions and within bounds."""

    @pytest.fixture
    def config(self):
        return NestConfig(placement_type="gravity", curve_tolerance=0.5)

    def test_no_duplicate_positions(self, config):
        """Parts should not be placed at identical positions."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"part-{i}",
                "polygons": [P((0, 0), (4, 0), (4, 3), (0, 3))],
            }
            for i in range(14)
        ]
        sheet = P((0, 0), (50, 0), (50, 30), (0, 30))
        rotations = [0.0] * 14

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        positions = {}
        for p in result.placements:
            pos_key = (round(p.x, 2), round(p.y, 2))
            if pos_key not in positions:
                positions[pos_key] = []
            positions[pos_key].append(p.uid)

        for pos, uids in positions.items():
            assert len(uids) == 1, f"Multiple parts at position {pos}: {uids}"

    def test_parts_within_sheet_bounds(self, config):
        """All placed parts should be within sheet bounds."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"part-{i}",
                "polygons": [P((0, 0), (5, 0), (5, 5), (0, 5))],
            }
            for i in range(10)
        ]
        sheet = P((0, 0), (50, 0), (50, 30), (0, 30))
        rotations = [0.0] * 10

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for p in result.placements:
            for poly in p.polygons:
                bounds = [
                    min(pt[0] for pt in poly),
                    min(pt[1] for pt in poly),
                    max(pt[0] for pt in poly),
                    max(pt[1] for pt in poly),
                ]
                assert bounds[0] >= -0.1, (
                    f"Part {p.uid} extends left of sheet: x={bounds[0]}"
                )
                assert bounds[1] >= -0.1, (
                    f"Part {p.uid} extends below sheet: y={bounds[1]}"
                )
                assert bounds[2] <= 50.1, (
                    f"Part {p.uid} extends right of sheet: x={bounds[2]}"
                )
                assert bounds[3] <= 30.1, (
                    f"Part {p.uid} extends above sheet: y={bounds[3]}"
                )

    def test_parts_touching_not_overlapping(self, config):
        """Parts can touch but should not overlap."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"rect-{i}",
                "polygons": [P((0, 0), (10, 0), (10, 10), (0, 10))],
            }
            for i in range(4)
        ]
        sheet = P((0, 0), (40, 0), (40, 40), (0, 40))
        rotations = [0.0] * 4

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) == 4

        # Check no overlaps (allowing touching)
        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(
                            poly1, poly2, min_overlap=0.01
                        )
                        assert overlap == 0, (
                            f"Parts {p1.uid} and {p2.uid} overlap by {overlap}"
                        )


class TestIFPCorrectness:
    """Test that Inner Fit Polygon is calculated correctly."""

    @pytest.fixture
    def config(self):
        return NestConfig(placement_type="gravity", curve_tolerance=0.5)

    def test_first_part_at_origin(self, config):
        """First part should be placed at or near origin (0, 0)."""
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "first",
                "polygons": [P((0, 0), (10, 0), (10, 10), (0, 10))],
            },
        ]
        sheet = P((0, 0), (100, 0), (100, 100), (0, 100))
        rotations = [0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        p = result.placements[0]
        assert p.x >= -0.1 and p.x <= 0.1, f"First part x={p.x}, expected ~0"
        assert p.y >= -0.1 and p.y <= 0.1, f"First part y={p.y}, expected ~0"


class TestStressTests:
    """Stress tests for larger numbers of parts."""

    @pytest.fixture
    def config(self):
        return NestConfig(placement_type="gravity", curve_tolerance=0.5)

    def test_20_identical_parts_no_overlap(self, config):
        """20 identical parts should all be placed without overlap."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"part-{i}",
                "polygons": [P((0, 0), (5, 0), (5, 4), (0, 4))],
            }
            for i in range(20)
        ]
        sheet = P((0, 0), (50, 0), (50, 40), (0, 40))
        rotations = [0.0] * 20

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) == 20

        # Verify no overlaps
        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Parts {p1.uid} and {p2.uid} overlap by {overlap}"
                        )

    def test_various_sizes_stress(self, config):
        """Mix of different sized parts should not overlap."""
        parts = []
        # 2 large parts
        for i in range(2):
            parts.append(
                {
                    "id": i,
                    "source": i,
                    "uid": f"large-{i}",
                    "polygons": [P((0, 0), (8, 0), (8, 8), (0, 8))],
                }
            )
        # 5 medium parts
        for i in range(5):
            parts.append(
                {
                    "id": 2 + i,
                    "source": 2 + i,
                    "uid": f"medium-{i}",
                    "polygons": [P((0, 0), (5, 0), (5, 5), (0, 5))],
                }
            )
        # 8 small parts
        for i in range(8):
            parts.append(
                {
                    "id": 7 + i,
                    "source": 7 + i,
                    "uid": f"small-{i}",
                    "polygons": [P((0, 0), (3, 0), (3, 3), (0, 3))],
                }
            )

        sheet = P((0, 0), (60, 0), (60, 40), (0, 40))
        rotations = [0.0] * 15

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        # Verify no overlaps
        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Parts {p1.uid} and {p2.uid} overlap by {overlap}"
                        )

    def test_mixed_aspect_ratios(self, config):
        """Test parts with very different aspect ratios."""
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "long-horizontal",
                "polygons": [P((0, 0), (100, 0), (100, 10), (0, 10))],
            },
            {
                "id": 1,
                "source": 1,
                "uid": "tall-vertical",
                "polygons": [P((0, 0), (10, 0), (10, 100), (0, 100))],
            },
            {
                "id": 2,
                "source": 2,
                "uid": "square",
                "polygons": [P((0, 0), (30, 0), (30, 30), (0, 30))],
            },
        ]
        sheet = P((0, 0), (200, 0), (200, 200), (0, 200))
        rotations = [0.0, 0.0, 0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0

    def test_multi_polygon_complex_part(self, config):
        """Test parts composed of multiple polygons (e.g., with holes)."""
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "part-with-hole",
                "polygons": [
                    P((0, 0), (40, 0), (40, 40), (0, 40)),
                    P((10, 10), (30, 10), (30, 30), (10, 30)),
                ],
            },
            {
                "id": 1,
                "source": 1,
                "uid": "simple-rect",
                "polygons": [P((0, 0), (20, 0), (20, 20), (0, 20))],
            },
        ]
        sheet = P((0, 0), (100, 0), (100, 100), (0, 100))
        rotations = [0.0, 0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(
                            poly1, poly2, min_overlap=0.5
                        )
                        assert overlap == 0


class TestComplexPolygons:
    """Test with complex polygons similar to real SVG imports."""

    @pytest.fixture
    def config(self):
        return NestConfig(
            placement_type="gravity",
            curve_tolerance=0.5,
            rotations=4,
            spacing=0.0,
        )

    def test_complex_star_shapes_no_overlap(self, config):
        """Star-shaped polygons should not overlap when nested."""
        import math

        def create_star(cx, cy, outer_r, inner_r, points):
            polygon = []
            for i in range(points * 2):
                angle = math.pi / 2 + (i * math.pi / points)
                r = outer_r if i % 2 == 0 else inner_r
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                polygon.append((x, y))
            return polygon

        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"star-{i}",
                "polygons": [create_star(5, 5, 5, 2, 5)],
            }
            for i in range(8)
        ]
        sheet = P((0, 0), (50, 0), (50, 50), (0, 50))
        rotations = [0.0] * 8

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) == 8

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Stars {p1.uid} and {p2.uid} overlap by {overlap}"
                        )

    def test_complex_l_shapes_no_overlap(self, config):
        """L-shaped polygons should not overlap when nested."""
        l_shape = [
            (0, 0),
            (10, 0),
            (10, 4),
            (4, 4),
            (4, 10),
            (0, 10),
        ]

        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"l-shape-{i}",
                "polygons": [l_shape.copy()],
            }
            for i in range(6)
        ]
        sheet = P((0, 0), (60, 0), (60, 40), (0, 40))
        rotations = [0.0, 0.0, 0.0, 90.0, 90.0, 90.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"L-shapes {p1.uid} and {p2.uid} "
                            f"overlap by {overlap}"
                        )

    def test_tightly_fitting_parts(self, config):
        """Test parts that should fit tightly together."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"tight-{i}",
                "polygons": [P((0, 0), (8, 0), (8, 6), (0, 6))],
            }
            for i in range(20)
        ]
        sheet = P((0, 0), (40, 0), (40, 30), (0, 30))
        rotations = [0.0] * 20

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        placed_count = len(result.placements)
        assert placed_count >= 15, f"Only {placed_count}/20 parts placed"

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Parts {p1.uid} and {p2.uid} overlap by {overlap}"
                        )

    def test_rotated_complex_polygons(self, config):
        """Complex polygons with various rotations should not overlap."""
        hexagon = [
            (5.0, 0.0),
            (10.0, 2.5),
            (10.0, 7.5),
            (5.0, 10.0),
            (0.0, 7.5),
            (0.0, 2.5),
        ]

        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"hex-{i}",
                "polygons": [hexagon.copy()],
            }
            for i in range(10)
        ]
        sheet = P((0, 0), (60, 0), (60, 50), (0, 50))
        rotations = [float(i * 60) for i in range(10)]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Hexagons {p1.uid} and {p2.uid} "
                            f"overlap by {overlap}"
                        )


class TestRealWorldAppScenario:
    """
    Test that replicates the exact scenario from the app.

    The app:
    1. Gets world geometry from workpieces
    2. Converts to polygons with curve_tolerance
    3. Applies spacing offset
    4. Normalizes polygons (subtracts min_x, min_y)
    5. Passes to DeepNest
    """

    @pytest.fixture
    def config(self):
        return NestConfig(
            placement_type="gravity",
            curve_tolerance=0.5,
            spacing=0.0,
            rotations=4,
        )

    def test_realistic_workflow_no_overlap(self, config):
        """
        Simulate the full workflow from the app:
        - Multiple parts with different shapes
        - Parts normalized to origin
        - Various rotations applied
        """
        import math

        def create_rounded_rect(width, height, corner_radius, num_pts=4):
            pts = []
            steps_per_corner = max(1, num_pts // 4)
            for corner in range(4):
                cx, cy, start_angle = {
                    0: (width - corner_radius, corner_radius, -math.pi / 2),
                    1: (width - corner_radius, height - corner_radius, 0),
                    2: (corner_radius, height - corner_radius, math.pi / 2),
                    3: (corner_radius, corner_radius, math.pi),
                }[corner]
                for i in range(steps_per_corner + 1):
                    angle = start_angle + i * (math.pi / 2) / steps_per_corner
                    x = cx + corner_radius * math.cos(angle)
                    y = cy + corner_radius * math.sin(angle)
                    pts.append((round(x, 3), round(y, 3)))
            return pts

        parts = []
        for i in range(12):
            part_poly = create_rounded_rect(8, 6, 1, num_pts=16)
            parts.append(
                {
                    "id": i,
                    "source": i,
                    "uid": f"rounded-rect-{i}",
                    "polygons": [part_poly],
                }
            )

        sheet = P((0, 0), (50, 0), (50, 40), (0, 40))
        rotations = [0.0] * 12

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Parts {p1.uid} and {p2.uid} overlap by {overlap}"
                        )

    def test_densly_packed_rectangles(self, config):
        """
        Test with many rectangles that should pack densely.
        This tests the core overlap detection under pressure.
        """
        parts = []
        for i in range(30):
            parts.append(
                {
                    "id": i,
                    "source": i,
                    "uid": f"rect-{i}",
                    "polygons": [P((0, 0), (5, 0), (5, 3), (0, 3))],
                }
            )

        sheet = P((0, 0), (40, 0), (40, 30), (0, 30))
        rotations = [0.0] * 30

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        assert len(result.placements) >= 24, (
            f"Expected at least 24 placements, got {len(result.placements)}"
        )

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Parts {p1.uid} and {p2.uid} overlap by {overlap}"
                        )

    def test_mixed_sizes_with_rotations(self, config):
        """Mix of sizes with various rotations - stress test."""
        parts = []

        for i in range(3):
            parts.append(
                {
                    "id": i,
                    "source": i,
                    "uid": f"large-{i}",
                    "polygons": [P((0, 0), (12, 0), (12, 8), (0, 8))],
                }
            )

        for i in range(5):
            parts.append(
                {
                    "id": 3 + i,
                    "source": 3 + i,
                    "uid": f"medium-{i}",
                    "polygons": [P((0, 0), (6, 0), (6, 4), (0, 4))],
                }
            )

        for i in range(10):
            parts.append(
                {
                    "id": 8 + i,
                    "source": 8 + i,
                    "uid": f"small-{i}",
                    "polygons": [P((0, 0), (3, 0), (3, 2), (0, 2))],
                }
            )

        sheet = P((0, 0), (50, 0), (50, 35), (0, 35))
        rotations = [0.0] * 18

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Parts {p1.uid} and {p2.uid} overlap by {overlap}"
                        )


class TestDeepNestIntegration:
    """
    Test the full DeepNest integration path that matches the app workflow.
    These tests use the DeepNest class directly with Geometry objects.
    """

    @pytest.fixture
    def config(self):
        return NestConfig(
            placement_type="gravity",
            curve_tolerance=0.5,
            spacing=0.0,
            rotations=4,
        )

    def test_deepnest_basic_no_overlap(self, config):
        """Test DeepNest class directly with basic polygons."""
        from rayforge.builtin_addons.deepnest.deepnest.deepnest import DeepNest
        from rayforge.core.geo import Geometry

        nester = DeepNest(config)

        for i in range(5):
            geo = Geometry()
            geo.move_to(0, 0)
            geo.line_to(10, 0)
            geo.line_to(10, 8)
            geo.line_to(0, 8)
            geo.close_path()
            nester.add_geometry(geo, uid=f"rect-{i}")

        solution = nester.nest()
        assert solution is not None
        assert len(solution.placements) == 5

        placements = solution.placements
        for i, p1 in enumerate(placements):
            for j, p2 in enumerate(placements):
                if i >= j:
                    continue
                for poly1 in p1["polygons"]:
                    for poly2 in p2["polygons"]:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0


class TestProblematicPolygonCases:
    """
    Test cases that might expose edge cases or bugs in the overlap detection.
    These simulate problematic scenarios from real-world SVG imports.
    """

    @pytest.fixture
    def config(self):
        return NestConfig(
            placement_type="gravity",
            curve_tolerance=0.5,
            spacing=0.0,
            rotations=4,
        )

    def test_many_vertices_polygon_no_overlap(self, config):
        """Test with polygons having many vertices (like traced bitmaps)."""
        import math

        def create_circle_polygon(cx, cy, radius, num_vertices):
            pts = []
            for i in range(num_vertices):
                angle = 2 * math.pi * i / num_vertices
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                pts.append((round(x, 3), round(y, 3)))
            return pts

        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"circle-{i}",
                "polygons": [create_circle_polygon(5, 5, 5, 32)],
            }
            for i in range(10)
        ]
        sheet = P((0, 0), (60, 0), (60, 50), (0, 50))
        rotations = [0.0] * 10

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0, (
                            f"Circles {p1.uid} and {p2.uid} "
                            f"overlap by {overlap}"
                        )

    def test_similar_sized_parts_no_overlap(self, config):
        """
        Test with parts that have very similar sizes.
        This tests for numerical precision issues.
        """
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "a",
                "polygons": [P((0, 0), (3.95, 0), (3.95, 3.07), (0, 3.07))],
            },
            {
                "id": 1,
                "source": 1,
                "uid": "b",
                "polygons": [P((0, 0), (3.83, 0), (3.83, 2.89), (0, 2.89))],
            },
            {
                "id": 2,
                "source": 2,
                "uid": "c",
                "polygons": [P((0, 0), (3.95, 0), (3.95, 3.07), (0, 3.07))],
            },
            {
                "id": 3,
                "source": 3,
                "uid": "d",
                "polygons": [P((0, 0), (3.83, 0), (3.83, 2.89), (0, 2.89))],
            },
        ]
        sheet = P((0, 0), (20, 0), (20, 15), (0, 15))
        rotations = [0.0, 0.0, 90.0, 90.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0

    def test_thin_elongated_parts_no_overlap(self, config):
        """Test with thin elongated parts that might cause precision issues."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"thin-{i}",
                "polygons": [P((0, 0), (15, 0), (15, 1), (0, 1))],
            }
            for i in range(10)
        ]
        sheet = P((0, 0), (50, 0), (50, 20), (0, 20))
        rotations = [0.0, 90.0] * 5

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0

    def test_nearly_touching_parts_no_overlap(self, config):
        """Test with parts that are nearly touching after placement."""
        parts = []
        for i in range(8):
            parts.append(
                {
                    "id": i,
                    "source": i,
                    "uid": f"near-{i}",
                    "polygons": [
                        P((0, 0), (5.01, 0), (5.01, 4.99), (0, 4.99))
                    ],
                }
            )

        sheet = P((0, 0), (30, 0), (30, 25), (0, 25))
        rotations = [0.0] * 8

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0

    def test_concave_polygons_no_overlap(self, config):
        """Test with concave polygons that might have intersection issues."""
        arrow = [
            (0, 5),
            (5, 5),
            (5, 0),
            (10, 7.5),
            (5, 15),
            (5, 10),
            (0, 10),
        ]

        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"arrow-{i}",
                "polygons": [arrow.copy()],
            }
            for i in range(6)
        ]
        sheet = P((0, 0), (60, 0), (60, 50), (0, 50))
        rotations = [0.0] * 6

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0

    def test_irregular_polygon_no_overlap(self, config):
        """Test with irregular polygons that might have edge cases."""
        irregular = [
            (0, 0),
            (8.3, 0.5),
            (12.1, 2.3),
            (10.7, 6.8),
            (7.2, 8.9),
            (3.1, 7.5),
            (0.8, 4.2),
        ]

        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"irreg-{i}",
                "polygons": [irregular.copy()],
            }
            for i in range(5)
        ]
        sheet = P((0, 0), (50, 0), (50, 40), (0, 40))
        rotations = [0.0, 72.0, 144.0, 216.0, 288.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0


class TestPlacementValidation:
    """
    Tests that validate the placement algorithm correctly rejects
    invalid placements.
    """

    @pytest.fixture
    def config(self):
        return NestConfig(
            placement_type="gravity",
            curve_tolerance=0.5,
            spacing=0.0,
        )

    def test_part_larger_than_sheet_not_placed(self, config):
        """A part larger than the sheet should not be placed."""
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "huge",
                "polygons": [P((0, 0), (100, 0), (100, 100), (0, 100))],
            }
        ]
        sheet = P((0, 0), (50, 0), (50, 50), (0, 50))
        rotations = [0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is None or len(result.placements) == 0

    def test_too_many_parts_for_space(self, config):
        """More parts than can fit should result in partial placement."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"big-{i}",
                "polygons": [P((0, 0), (20, 0), (20, 20), (0, 20))],
            }
            for i in range(10)
        ]
        sheet = P((0, 0), (40, 0), (40, 40), (0, 40))
        rotations = [0.0] * 10

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) < 10

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0

    def test_placement_bounds_respected(self, config):
        """All placements should be within sheet bounds."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"part-{i}",
                "polygons": [P((0, 0), (8, 0), (8, 6), (0, 6))],
            }
            for i in range(10)
        ]
        sheet = P((0, 0), (35, 0), (35, 25), (0, 25))
        rotations = [0.0] * 10

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for p in result.placements:
            for poly in p.polygons:
                min_x = min(pt[0] for pt in poly)
                min_y = min(pt[1] for pt in poly)
                max_x = max(pt[0] for pt in poly)
                max_y = max(pt[1] for pt in poly)

                assert min_x >= -0.01, f"Part {p.uid} extends left of sheet"
                assert min_y >= -0.01, f"Part {p.uid} extends below sheet"
                assert max_x <= 35.01, f"Part {p.uid} extends right of sheet"
                assert max_y <= 25.01, f"Part {p.uid} extends above sheet"

    def test_touching_corners_no_false_positive(self, config):
        """Parts touching at corners should not be detected as overlapping."""
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "bottom-left",
                "polygons": [P((0, 0), (10, 0), (10, 10), (0, 10))],
            },
            {
                "id": 1,
                "source": 1,
                "uid": "top-right",
                "polygons": [P((10, 10), (20, 10), (20, 20), (10, 20))],
            },
        ]
        sheet = P((0, 0), (50, 0), (50, 50), (0, 50))
        rotations = [0.0, 0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0

    def test_slightly_overlapping_parts_detected(self, config):
        """Parts that slightly overlap should be detected and not placed."""
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "left",
                "polygons": [P((0, 0), (10.1, 0), (10.1, 10), (0, 10))],
            },
            {
                "id": 1,
                "source": 1,
                "uid": "right",
                "polygons": [P((10, 0), (20, 0), (20, 10), (10, 10))],
            },
        ]
        sheet = P((0, 0), (50, 0), (50, 50), (0, 50))
        rotations = [0.0, 0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(
                            poly1, poly2, min_overlap=0.5
                        )
                        assert overlap == 0, (
                            f"Parts should not overlap, but found {overlap}"
                        )


class TestGeneticAlgorithmRotations:
    """
    Test that various rotation combinations from the genetic algorithm
    don't cause overlap issues.
    """

    @pytest.fixture
    def config(self):
        return NestConfig(
            placement_type="gravity",
            curve_tolerance=0.5,
            spacing=0.0,
            rotations=4,
        )

    def test_random_rotations_no_overlap(self, config):
        """Test with random rotation combinations."""
        import random

        random.seed(42)

        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"rect-{i}",
                "polygons": [P((0, 0), (8, 0), (8, 5), (0, 5))],
            }
            for i in range(12)
        ]
        sheet = P((0, 0), (50, 0), (50, 40), (0, 40))

        for _ in range(10):
            rotations = [
                random.choice([0.0, 90.0, 180.0, 270.0]) for _ in range(12)
            ]

            result = place_parts(parts, as_sheets([sheet]), rotations, config)
            assert result is not None

            for i, p1 in enumerate(result.placements):
                for j, p2 in enumerate(result.placements):
                    if i >= j:
                        continue
                    for poly1 in p1.polygons:
                        for poly2 in p2.polygons:
                            overlap = polygons_intersect(poly1, poly2)
                            assert overlap == 0, (
                                f"Parts {p1.uid} (rot={p1.rotation}) and "
                                f"{p2.uid} (rot={p2.rotation}) "
                                f"overlap by {overlap}"
                            )

    def test_all_parts_rotated_90(self, config):
        """Test with all parts rotated 90 degrees."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"rect-{i}",
                "polygons": [P((0, 0), (8, 0), (8, 5), (0, 5))],
            }
            for i in range(8)
        ]
        sheet = P((0, 0), (50, 0), (50, 40), (0, 40))
        rotations = [90.0] * 8

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0

    def test_alternating_rotations_no_overlap(self, config):
        """Test with alternating 0/90 degree rotations."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"rect-{i}",
                "polygons": [P((0, 0), (10, 0), (10, 6), (0, 6))],
            }
            for i in range(10)
        ]
        sheet = P((0, 0), (50, 0), (50, 35), (0, 35))
        rotations = [0.0 if i % 2 == 0 else 90.0 for i in range(10)]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0

    def test_mixed_rotations_with_l_shapes(self, config):
        """Test L-shapes with mixed rotations."""
        l_shape = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]

        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"l-{i}",
                "polygons": [l_shape.copy()],
            }
            for i in range(8)
        ]
        sheet = P((0, 0), (60, 0), (60, 50), (0, 50))
        rotations = [0.0, 90.0, 180.0, 270.0] * 2

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0


class TestNonRectangularSheet:
    """Test nesting on non-rectangular sheets."""

    @pytest.fixture
    def config(self):
        return NestConfig(
            placement_type="gravity",
            curve_tolerance=0.5,
            spacing=0.0,
        )

    def test_polygonal_sheet_no_overlap(self, config):
        """Test nesting on a polygonal (non-rectangular) sheet."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"rect-{i}",
                "polygons": [P((0, 0), (5, 0), (5, 4), (0, 4))],
            }
            for i in range(6)
        ]

        sheet = P((0, 0), (30, 0), (40, 20), (30, 40), (0, 40))
        rotations = [0.0] * 6

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        for i, p1 in enumerate(result.placements):
            for j, p2 in enumerate(result.placements):
                if i >= j:
                    continue
                for poly1 in p1.polygons:
                    for poly2 in p2.polygons:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0

    def test_sheet_with_offset_origin(self, config):
        """Test nesting on a sheet that doesn't start at origin."""
        from rayforge.builtin_addons.deepnest.deepnest.deepnest import DeepNest
        from rayforge.core.geo import Geometry

        nester = DeepNest(config)

        sheet = Geometry()
        sheet.move_to(20, 20)
        sheet.line_to(70, 20)
        sheet.line_to(70, 60)
        sheet.line_to(20, 60)
        sheet.close_path()
        nester.add_geometry(sheet, uid="sheet", is_sheet=True)

        for i in range(4):
            geo = Geometry()
            geo.move_to(0, 0)
            geo.line_to(10, 0)
            geo.line_to(10, 8)
            geo.line_to(0, 8)
            geo.close_path()
            nester.add_geometry(geo, uid=f"rect-{i}")

        solution = nester.nest()
        assert solution is not None

        placements = solution.placements
        for i, p1 in enumerate(placements):
            for j, p2 in enumerate(placements):
                if i >= j:
                    continue
                for poly1 in p1["polygons"]:
                    for poly2 in p2["polygons"]:
                        overlap = polygons_intersect(poly1, poly2)
                        assert overlap == 0


class TestTransformApplication:
    """
    Tests that verify the transformation logic matches what the placement
    algorithm expects. This tests the same logic used in layout_cmd.py.
    """

    @pytest.fixture
    def config(self):
        return NestConfig(placement_type="gravity", curve_tolerance=0.5)

    def _rotate_polygon(self, polygon, angle_deg):
        """Rotate a polygon around the origin."""
        import math

        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        result = []
        for x, y in polygon:
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            result.append((new_x, new_y))
        return result

    def _translate_polygon(self, polygon, dx, dy):
        """Translate a polygon."""
        return [(x + dx, y + dy) for x, y in polygon]

    def _polygon_bounds(self, polygon):
        """Get the bounding box of a polygon."""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    def _apply_placement_transform(self, original_polygon, placement):
        """
        Apply the placement transform to a polygon the way layout_cmd.py does.

        The transform is T @ R @ old_world where:
        - R = rotation
        - T = placement_pos - min(R @ world_geo)
        - old_world is implicit (we work directly with world coordinates)

        This gives the same result as the placement algorithm:
        final_geo = T @ R @ world_geo
        where min(final_geo) = placement_pos
        """
        import math

        placement_x = placement.x
        placement_y = placement.y
        rotation = placement.rotation

        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rotated_points = [
            (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
            for x, y in original_polygon
        ]
        rotated_min_x = min(p[0] for p in rotated_points)
        rotated_min_y = min(p[1] for p in rotated_points)

        tx = placement_x - rotated_min_x
        ty = placement_y - rotated_min_y

        translated = self._translate_polygon(rotated_points, tx, ty)
        return translated

    def test_transform_matches_placement_expectation_no_rotation(self, config):
        """
        When rotation is 0, the transform should place the polygon at the
        placement position.
        """
        from rayforge.builtin_addons.deepnest.deepnest.deepnest.placement import (  # noqa: E501
            place_parts,
        )

        sheet = P((0, 0), (50, 0), (50, 50), (0, 50))
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "rect",
                "polygons": [P((5, 5), (15, 5), (15, 15), (5, 15))],
            }
        ]
        rotations = [0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) == 1

        p = result.placements[0]
        original = P((5, 5), (15, 5), (15, 15), (5, 15))

        transformed = self._apply_placement_transform(original, p)

        for i, (tx, ty) in enumerate(transformed):
            px, py = p.polygons[0][i]
            assert abs(tx - px) < 0.01, (
                f"X mismatch at point {i}: {tx} vs {px}"
            )
            assert abs(ty - py) < 0.01, (
                f"Y mismatch at point {i}: {ty} vs {py}"
            )

    def test_transform_matches_placement_expectation_with_rotation(
        self, config
    ):
        """
        When rotation is non-zero, the transform should still match.
        """
        from rayforge.builtin_addons.deepnest.deepnest.deepnest.placement import (  # noqa: E501
            place_parts,
        )

        sheet = P((0, 0), (50, 0), (50, 50), (0, 50))
        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "rect",
                "polygons": [P((5, 5), (15, 5), (15, 15), (5, 15))],
            }
        ]
        rotations = [90.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) == 1

        p = result.placements[0]
        original = P((5, 5), (15, 5), (15, 15), (5, 15))

        transformed = self._apply_placement_transform(original, p)

        for i, (tx, ty) in enumerate(transformed):
            px, py = p.polygons[0][i]
            assert abs(tx - px) < 0.01, (
                f"X mismatch at point {i}: {tx} vs {px}"
            )
            assert abs(ty - py) < 0.01, (
                f"Y mismatch at point {i}: {ty} vs {py}"
            )

    def test_multiple_parts_with_rotation_no_overlap_after_transform(
        self, config
    ):
        """
        Test that after applying transforms to multiple parts with rotations,
        there are no overlaps.
        """
        from rayforge.builtin_addons.deepnest.deepnest.deepnest.placement import (  # noqa: E501
            place_parts,
        )

        sheet = P((0, 0), (50, 0), (50, 50), (0, 50))
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"rect-{i}",
                "polygons": [P((3, 3), (13, 3), (13, 10), (3, 10))],
            }
            for i in range(6)
        ]
        rotations = [0.0, 90.0, 180.0, 270.0, 0.0, 90.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None

        original_polygon = P((3, 3), (13, 3), (13, 10), (3, 10))
        transformed_polygons = []

        for p in result.placements:
            transformed = self._apply_placement_transform(original_polygon, p)
            transformed_polygons.append(transformed)

        for i, poly1 in enumerate(transformed_polygons):
            for j, poly2 in enumerate(transformed_polygons):
                if i >= j:
                    continue
                overlap = polygons_intersect(poly1, poly2)
                assert overlap == 0, (
                    f"Transformed polygons {i} and {j} "
                    f"overlap with area {overlap}"
                )

    def test_geometry_at_non_origin_position(self, config):
        """
        Test geometry at non-origin position (like real workpieces
        where world geometry min might be at (60, 47) not (0, 0)).
        """
        from rayforge.builtin_addons.deepnest.deepnest.deepnest.placement import (  # noqa: E501
            place_parts,
        )

        sheet = P((0, 0), (50, 0), (50, 50), (0, 50))

        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"rect-{i}",
                "polygons": [
                    P(
                        (50 + i * 5, 40 + i * 3),
                        (60 + i * 5, 40 + i * 3),
                        (60 + i * 5, 50 + i * 3),
                        (50 + i * 5, 50 + i * 3),
                    )
                ],
            }
            for i in range(4)
        ]
        rotations = [0.0, 90.0, 180.0, 270.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) == 4

        for i, p in enumerate(result.placements):
            original = parts[i]["polygons"][0]
            transformed = self._apply_placement_transform(original, p)

            for j, (tx, ty) in enumerate(transformed):
                px, py = p.polygons[0][j]
                assert abs(tx - px) < 0.01, (
                    f"Part {i}, point {j}: X mismatch {tx} vs {px}"
                )
                assert abs(ty - py) < 0.01, (
                    f"Part {i}, point {j}: Y mismatch {ty} vs {py}"
                )

        transformed_polygons = [
            self._apply_placement_transform(parts[i]["polygons"][0], p)
            for i, p in enumerate(result.placements)
        ]

        for i, poly1 in enumerate(transformed_polygons):
            for j, poly2 in enumerate(transformed_polygons):
                if i >= j:
                    continue
                overlap = polygons_intersect(poly1, poly2)
                assert overlap == 0, (
                    f"Transformed polygons {i} and {j} "
                    f"overlap with area {overlap}"
                )

    def test_with_non_uniform_scale(self, config):
        """
        Test with non-uniform scale to match real workpieces.
        The geometry is already scaled (like world geometry from
        get_world_geometry()), and we need to preserve that scale.
        """
        from rayforge.builtin_addons.deepnest.deepnest.deepnest.placement import (  # noqa: E501
            place_parts,
        )

        sheet = P((0, 0), (200, 0), (200, 200), (0, 200))

        scale_x, scale_y = 3.94, 7.88

        base_rect = P((0, 0), (10, 0), (10, 10), (0, 10))
        scaled_rect = [(x * scale_x, y * scale_y) for x, y in base_rect]

        world_offset_x, world_offset_y = 60.0, 47.0
        world_rect = [
            (x + world_offset_x, y + world_offset_y) for x, y in scaled_rect
        ]

        parts = [
            {
                "id": 0,
                "source": 0,
                "uid": "scaled-rect",
                "polygons": [world_rect],
            }
        ]
        rotations = [0.0]

        result = place_parts(parts, as_sheets([sheet]), rotations, config)
        assert result is not None
        assert len(result.placements) == 1

        p = result.placements[0]

        cos_a, sin_a = 1.0, 0.0
        rotated = [
            (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
            for x, y in world_rect
        ]
        rotated_min_x = min(pt[0] for pt in rotated)
        rotated_min_y = min(pt[1] for pt in rotated)

        tx = p.x - rotated_min_x
        ty = p.y - rotated_min_y

        transformed = [(x + tx, y + ty) for x, y in world_rect]

        for i, ((tx_val, ty_val), (px, py)) in enumerate(
            zip(transformed, p.polygons[0])
        ):
            assert abs(tx_val - px) < 0.01, (
                f"Point {i}: X mismatch {tx_val} vs {px}"
            )
            assert abs(ty_val - py) < 0.01, (
                f"Point {i}: Y mismatch {ty_val} vs {py}"
            )
