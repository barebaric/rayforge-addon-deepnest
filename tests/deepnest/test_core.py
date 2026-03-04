"""
Tests for rayforge.shared.deepnest.core module.
"""

import pytest
from rayforge.core.geo import Geometry
from deepnest.deepnest.models import (
    NestConfig,
    WorkpieceInfo,
)
from deepnest.deepnest.core import (
    DeepNest,
    nest_geometries,
)


class TestWorkpieceInfo:
    def test_default(self):
        info = WorkpieceInfo(
            uid="test",
            polygons=[[(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]],
            source=0,
        )
        assert info.uid == "test"
        assert len(info.polygons) == 1
        assert info.source == 0
        assert info.quantity == 1
        assert info.is_sheet is False


class TestDeepNest:
    @pytest.fixture
    def config(self):
        return NestConfig(
            population_size=5,
            rotations=4,
            mutation_rate=10,
        )

    def test_init(self, config):
        nester = DeepNest(config)
        assert nester.config == config

    def test_init_default(self):
        nester = DeepNest()
        assert nester.config is not None

    def test_add_geometry(self, config):
        nester = DeepNest(config)

        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.line_to(5, 10)
        geo.close_path()

        assert nester.add_geometry(geo)

    def test_add_empty_geometry(self, config):
        nester = DeepNest(config)
        assert not nester.add_geometry(Geometry())

    def test_add_multi_segment_geometry(self, config):
        nester = DeepNest(config)

        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.line_to(10, 10)
        geo.line_to(0, 10)
        geo.close_path()
        geo.move_to(20, 0)
        geo.line_to(30, 0)
        geo.line_to(30, 10)
        geo.line_to(20, 10)
        geo.close_path()

        assert nester.add_geometry(geo)
        assert len(nester._workpieces) == 1
        assert len(nester._workpieces[0].polygons) == 2

    def test_nest_simple(self, config):
        nester = DeepNest(config)

        geo1 = Geometry()
        geo1.move_to(0, 0)
        geo1.line_to(10, 0)
        geo1.line_to(10, 10)
        geo1.line_to(0, 10)
        geo1.close_path()
        nester.add_geometry(geo1, uid="part1")

        geo2 = Geometry()
        geo2.move_to(0, 0)
        geo2.line_to(8, 0)
        geo2.line_to(8, 8)
        geo2.line_to(0, 8)
        geo2.close_path()
        nester.add_geometry(geo2, uid="part2")

        sheet = Geometry()
        sheet.move_to(0, 0)
        sheet.line_to(100, 0)
        sheet.line_to(100, 100)
        sheet.line_to(0, 100)
        sheet.close_path()
        nester.add_geometry(sheet, uid="sheet", is_sheet=True)

        result = nester.nest()

        assert result is not None
        assert len(result.placements) >= 1

    def test_nest_no_parts(self, config):
        nester = DeepNest(config)
        result = nester.nest()
        assert result is None

    def test_nest_cancel(self, config):
        nester = DeepNest(config)

        for i in range(5):
            geo = Geometry()
            geo.move_to(0, 0)
            geo.line_to(10 + i, 0)
            geo.line_to(5 + i, 10)
            geo.close_path()
            nester.add_geometry(geo, uid=f"part_{i}")

        nester.cancel()
        assert nester._cancelled

    def test_clear(self, config):
        nester = DeepNest(config)

        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.line_to(5, 10)
        geo.close_path()
        nester.add_geometry(geo)

        nester.clear()
        assert len(nester._workpieces) == 0


class TestNestFunctions:
    def test_nest_geometries_empty(self):
        result = nest_geometries([])
        assert result is None


class TestIntegration:
    @pytest.fixture
    def config(self):
        return NestConfig(
            population_size=3,
            rotations=2,
            curve_tolerance=0.5,
        )

    def test_nest_rectangles(self, config):
        nester = DeepNest(config)

        for i in range(3):
            geo = Geometry()
            size = 10 + i * 2
            geo.move_to(0, 0)
            geo.line_to(size, 0)
            geo.line_to(size, size)
            geo.line_to(0, size)
            geo.close_path()
            nester.add_geometry(geo, uid=f"rect_{i}")

        sheet = Geometry()
        sheet.move_to(0, 0)
        sheet.line_to(100, 0)
        sheet.line_to(100, 50)
        sheet.line_to(0, 50)
        sheet.close_path()
        nester.add_geometry(sheet, uid="sheet", is_sheet=True)

        result = nester.nest()

        assert result is not None
        assert len(result.placements) >= 1

    def test_nest_triangles(self, config):
        nester = DeepNest(config)

        for i in range(4):
            geo = Geometry()
            base = 10 + i * 3
            geo.move_to(0, 0)
            geo.line_to(base, 0)
            geo.line_to(base / 2, base)
            geo.close_path()
            nester.add_geometry(geo, uid=f"tri_{i}")

        sheet = Geometry()
        sheet.move_to(0, 0)
        sheet.line_to(100, 0)
        sheet.line_to(100, 100)
        sheet.line_to(0, 100)
        sheet.close_path()
        nester.add_geometry(sheet, uid="sheet", is_sheet=True)

        result = nester.nest()

        assert result is not None
        assert len(result.placements) >= 1

    def test_nest_multi_segment_geometry(self, config):
        nester = DeepNest(config)

        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.line_to(10, 10)
        geo.line_to(0, 10)
        geo.close_path()
        geo.move_to(15, 0)
        geo.line_to(25, 0)
        geo.line_to(25, 10)
        geo.line_to(15, 10)
        geo.close_path()
        nester.add_geometry(geo, uid="two_rects")

        sheet = Geometry()
        sheet.move_to(0, 0)
        sheet.line_to(100, 0)
        sheet.line_to(100, 100)
        sheet.line_to(0, 100)
        sheet.close_path()
        nester.add_geometry(sheet, uid="sheet", is_sheet=True)

        result = nester.nest()

        assert result is not None
        assert len(result.placements) == 1
        assert len(result.placements[0]["polygons"]) == 2
