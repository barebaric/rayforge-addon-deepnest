"""
Tests for nesting layout strategy, specifically focusing on stock geometry
handling with transformations (rotation, scale, shear).

Tests verify that:
1. Stock geometry is correctly transformed (rotation, scale, shear)
2. Nesting strategy uses correct geometry for rectangular vs non-rectangular
3. Placement positions correctly account for stock offset
4. Workpieces are placed within stock boundaries, not just in bbox
"""

from pathlib import Path

import pytest
from rayforge.core.geo.geometry import Geometry
from rayforge.core.matrix import Matrix
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.source_asset import SourceAsset
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.image import SVG_RENDERER

from deepnest.nesting import NestingLayoutStrategy
from deepnest.deepnest import Placement


def _create_workpiece_with_geometry(
    doc: Doc,
    geometry: Geometry,
    name: str = "test_wp",
    matrix: Matrix | None = None,
) -> WorkPiece:
    """
    Creates a WorkPiece with actual geometry for testing.

    The input geometry is assumed to be in world space. This function
    normalizes it to 1x1 space for the pristine_geometry and
    stores the normalization matrix.
    """
    source = SourceAsset(
        source_file=Path(f"{name}.svg"),
        original_data=b"<svg></svg>",
        renderer=SVG_RENDERER,
    )
    doc.add_asset(source)

    # Calculate normalization to 1x1 space
    geo_rect = geometry.rect()
    min_x, min_y, max_x, max_y = geo_rect
    width = max_x - min_x
    height = max_y - min_y

    # Create 1x1 pristine geometry (normalized)
    pristine_geo = Geometry()
    pristine_geo.move_to(0, 0)
    pristine_geo.line_to(1, 0)
    pristine_geo.line_to(1, 1)
    pristine_geo.line_to(0, 1)
    pristine_geo.close_path()

    # Normalization matrix scales from 1x1 to actual size
    norm_matrix = Matrix.scale(width, height)

    segment = SourceAssetSegment(
        source_asset_uid=source.uid,
        pristine_geometry=pristine_geo,
        normalization_matrix=norm_matrix,
        vectorization_spec=PassthroughSpec(),
    )

    wp = WorkPiece(name=name, source_segment=segment)

    layer = doc.active_layer
    if layer:
        layer.add_child(wp)

    if matrix is not None:
        wp.matrix = matrix
    else:
        # Set the workpiece matrix to position it correctly
        wp.matrix = Matrix.translation(min_x, min_y)

    return wp


def _create_stock_with_geometry(
    doc: Doc,
    geometry: Geometry,
    matrix: Matrix | None = None,
) -> StockItem:
    """
    Creates a StockItem with actual geometry for testing.
    """
    asset = StockAsset(name="Test Stock", geometry=geometry)
    doc.add_asset(asset)

    stock = StockItem(stock_asset_uid=asset.uid, name="Test Stock")
    if matrix is not None:
        stock.matrix = matrix
    doc.add_child(stock)

    return stock


class TestNestingLayoutStrategyStockGeometry:
    """Tests for _get_stock_geometry in NestingLayoutStrategy."""

    def test_stock_geometry_in_world_space(self):
        """Test that stock geometry is returned in world space."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(100, 0)
        geo.line_to(100, 50)
        geo.line_to(0, 50)
        geo.close_path()

        matrix = Matrix.scale(100, 50)
        matrix = Matrix.translation(50, 50) @ matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, geo, matrix)

        strategy = NestingLayoutStrategy([stock])
        result_geo = strategy._get_stock_geometry(stock)

        assert result_geo is not None
        assert not result_geo.is_empty()

        rect = result_geo.rect()
        # Geometry should be at world position (50, 50)
        assert rect[0] == pytest.approx(50)
        assert rect[1] == pytest.approx(50)

    def test_rotated_stock_geometry_in_world_space(self):
        """Test that rotated stock geometry is returned in world space."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(100, 0)
        geo.line_to(100, 50)
        geo.line_to(0, 50)
        geo.close_path()

        matrix = Matrix.rotation(45)
        matrix = Matrix.translation(200, 100) @ matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, geo, matrix)

        strategy = NestingLayoutStrategy([stock])
        result_geo = strategy._get_stock_geometry(stock)

        assert result_geo is not None
        assert not result_geo.is_empty()

        rect = result_geo.rect()
        # Geometry should be at world position (200, 100) with rotation
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        assert width > 1.0
        assert height > 0.5

    def test_non_rectangular_stock_geometry_in_world_space(self):
        """Test that non-rectangular (L-shaped) stock geometry works."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(100, 0)
        geo.line_to(100, 25)
        geo.line_to(25, 25)
        geo.line_to(25, 100)
        geo.line_to(0, 100)
        geo.close_path()

        matrix = Matrix.scale(100, 100)
        matrix = Matrix.translation(50, 50) @ matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, geo, matrix)

        strategy = NestingLayoutStrategy([stock])
        result_geo = strategy._get_stock_geometry(stock)

        assert result_geo is not None
        assert not result_geo.is_empty()

        rect = result_geo.rect()
        # Geometry should be at world position (50, 50)
        assert rect[0] == pytest.approx(50)
        assert rect[1] == pytest.approx(50)

    def test_stock_polygons_returns_correct_tuples(self):
        """Test that _get_stock_polygons returns (geometry, uid) tuples."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(100, 0)
        geo.line_to(100, 50)
        geo.line_to(0, 50)
        geo.close_path()

        matrix = Matrix.scale(100, 50)
        matrix = Matrix.translation(75, 25) @ matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, geo, matrix)

        strategy = NestingLayoutStrategy([stock])
        polygons = strategy._get_stock_polygons()

        assert len(polygons) == 1
        sheet_geo, sheet_uid = polygons[0]
        assert isinstance(sheet_geo, Geometry)
        assert sheet_uid == "stock-0"


class TestNestingStockGeometryTransformations:
    """Tests that reveal bugs in stock geometry transformations."""

    def test_world_geometry_with_translation_has_correct_size(self):
        """
        Test that stock world geometry has correct bounding box when
        translated.

        This test directly checks get_world_geometry return value, bypassing
        the nesting algorithm to isolate the bug.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(200, 100)
        stock_matrix = Matrix.translation(50, 50) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        # Get the world geometry that nesting.py will receive
        world_geo = stock.get_world_geometry()
        world_rect = world_geo.rect()

        # World geometry should be at (50, 50) to (250, 150)
        assert world_rect[0] == pytest.approx(50), (
            f"World rect min_x should be 50, got {world_rect[0]}"
        )
        assert world_rect[1] == pytest.approx(50), (
            f"World rect min_y should be 50, got {world_rect[1]}"
        )
        assert world_rect[2] == pytest.approx(250), (
            f"World rect max_x should be 250, got {world_rect[2]}"
        )
        assert world_rect[3] == pytest.approx(150), (
            f"World rect max_y should be 150, got {world_rect[3]}"
        )

        # The stock offset should be (50, 50)
        strategy = NestingLayoutStrategy([stock])
        sheet_geo, sheet_uid = strategy._get_stock_polygons()[0]

        # Sheet size should be 200x100
        sheet_rect = sheet_geo.rect()
        sheet_width = sheet_rect[2] - sheet_rect[0]
        sheet_height = sheet_rect[3] - sheet_rect[1]

        assert sheet_width == pytest.approx(200, rel=0.1), (
            f"Sheet width {sheet_width} should be 200 (not {sheet_width})"
        )
        assert sheet_height == pytest.approx(100, rel=0.1), (
            f"Sheet height {sheet_height} should be 100 (not {sheet_height})"
        )

        # Sheet geometry should be at world position (50, 50)
        assert sheet_rect[0] == pytest.approx(50)
        assert sheet_rect[1] == pytest.approx(50)

    def test_world_geometry_with_rotation_has_correct_size(self):
        """
        Test that stock world geometry has correct bounding box when rotated.

        BUG CHECK: Rotated stock geometry might be incorrect.

        For a 100x50 rectangle rotated 30 degrees:
        - Width = 100*cos(30) + 50*sin(30) = 86.6 + 25 = 111.6
        - Height = 100*sin(30) + 50*cos(30) = 50 + 43.3 = 93.3
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(100, 0)
        stock_geo.line_to(100, 50)
        stock_geo.line_to(0, 50)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(100, 50)
        stock_matrix = Matrix.rotation(30) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        world_geo = stock.get_world_geometry()
        world_rect = world_geo.rect()

        # Check that world geometry has expected bounding box for rotated
        # rectangle
        world_width = world_rect[2] - world_rect[0]
        world_height = world_rect[3] - world_rect[1]

        assert world_width == pytest.approx(111.6, rel=0.1), (
            f"World width {world_width} should be ~111.6 for "
            f"100x50 rotated 30°"
        )
        assert world_height == pytest.approx(93.3, rel=0.1), (
            f"World height {world_height} should be ~93.3 for "
            f"100x50 rotated 30°"
        )

    def test_world_geometry_with_scale_has_correct_size(self):
        """
        Test that stock world geometry has correct bounding box when scaled.

        BUG CHECK: Scaled stock geometry might be incorrect.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(400, 200)

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        world_geo = stock.get_world_geometry()
        world_rect = world_geo.rect()

        # World geometry should be 400x200 (200x100 scaled by 2)
        world_width = world_rect[2] - world_rect[0]
        world_height = world_rect[3] - world_rect[1]

        assert world_width == pytest.approx(400, rel=0.1), (
            f"World width {world_width} should be 400"
        )
        assert world_height == pytest.approx(200, rel=0.1), (
            f"World height {world_height} should be 200"
        )

    def test_world_geometry_with_shear_has_correct_size(self):
        """
        Test that stock world geometry has correct bounding box when sheared.

        BUG CHECK: Sheared stock geometry might be incorrect.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(100, 0)
        stock_geo.line_to(100, 50)
        stock_geo.line_to(0, 50)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(100, 50)
        stock_matrix = Matrix.shear(0.3, 0) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        world_geo = stock.get_world_geometry()
        world_rect = world_geo.rect()

        # Shear will change bbox, check it's reasonable
        world_width = world_rect[2] - world_rect[0]
        world_height = world_rect[3] - world_rect[1]

        assert world_width > 90 and world_width < 150, (
            f"World width {world_width} seems wrong for sheared 100x50 stock"
        )
        assert world_height > 40 and world_height < 80, (
            f"World height {world_height} seems wrong for sheared 100x50 stock"
        )


class TestNestingPlacementWithTransformedStock:
    """Tests for placement computation with transformed stock."""

    def test_translated_stock_with_empty_uses_unit_square(self):
        """
        Test that when StockAsset has no geometry, a 1x1 unit square
        is used as the base for stock.

        The StockAsset contract is that if no geometry is provided,
        it uses a 1x1 empty Geometry. This is the "natural" or "unit"
        size of the stock.

        BUG REVEALED: When stock.matrix includes translation (50, 50) and
        StockAsset has empty geometry (defaulting to 1x1), the
        _get_stock_geometry() method appears to apply stock.world_transform
        twice - once to create the unit square in get_world_rect_geometry(),
        then to normalize it (subtract offset).

        Test confirms: Sheet polygon passed to deepnest should be 1x1,
        offset should be (50, 50), resulting in effective stock at
        (50, 50) to (51, 51).
        """
        # Create stock WITHOUT explicit geometry (uses default empty
        # Geometry())
        # This triggers the StockAsset 1x1 unit square contract
        stock = StockItem(stock_asset_uid="", name="Test Stock")
        stock.matrix = Matrix.translation(50, 50)

        doc = Doc()
        doc.add_child(stock)

        # Get the stock polygons that would be passed to deepnest
        strategy = NestingLayoutStrategy([stock])
        stock_polygons = strategy._get_stock_polygons()

        assert len(stock_polygons) == 1
        sheet_geo, sheet_uid = stock_polygons[0]

        # Sheet geometry should be 1x1 (unit square from StockAsset)
        sheet_rect = sheet_geo.rect()
        sheet_width = sheet_rect[2] - sheet_rect[0]
        sheet_height = sheet_rect[3] - sheet_rect[1]

        assert sheet_width == pytest.approx(1.0, rel=0.01), (
            f"Sheet width {sheet_width} should be 1.0 "
            f"(unit square from StockAsset)"
        )
        assert sheet_height == pytest.approx(1.0, rel=0.01), (
            f"Sheet height {sheet_height} should be 1.0"
        )

        # Sheet geometry should be at world position (50, 50)
        sheet_rect = sheet_geo.rect()
        assert sheet_rect[0] == pytest.approx(50)
        assert sheet_rect[1] == pytest.approx(50)

    def test_rotated_stock_sheet_polygon_has_correct_size(self):
        """
        Test that sheet polygon passed to deepnest is correctly sized
        when stock is rotated.

        BUG: Rotated stock causes items to be placed way outside canvas.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(100, 0)
        stock_geo.line_to(100, 50)
        stock_geo.line_to(0, 50)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(100, 50)
        stock_matrix = Matrix.rotation(30) @ stock_matrix
        stock_matrix = Matrix.translation(100, 100) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(20, 0)
        wp_geo.line_to(20, 20)
        wp_geo.line_to(0, 20)
        wp_geo.close_path()

        wp1 = _create_workpiece_with_geometry(doc, wp_geo, "wp1")
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        # Get the stock polygons that would be passed to deepnest
        stock_polygons = strategy._get_stock_polygons()

        assert len(stock_polygons) == 1
        sheet_geo, sheet_uid = stock_polygons[0]

        # The sheet geometry should have reasonable size (rotated 100x50)
        sheet_rect = sheet_geo.rect()
        sheet_width = sheet_rect[2] - sheet_rect[0]
        sheet_height = sheet_rect[3] - sheet_rect[1]

        assert sheet_width > 50, f"Sheet width {sheet_width} should be > 50"
        assert sheet_height > 20, f"Sheet height {sheet_height} should be > 20"

    def test_non_rectangular_stock_sheet_has_correct_size(self):
        """
        Test that sheet polygon for L-shaped stock is correct.

        BUG: Non-rectangular stock causes items to be placed way
        outside canvas.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 25)
        stock_geo.line_to(25, 25)
        stock_geo.line_to(25, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(200, 100)
        stock_matrix = Matrix.translation(50, 50) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(15, 0)
        wp_geo.line_to(15, 15)
        wp_geo.line_to(0, 15)
        wp_geo.close_path()

        wp1 = _create_workpiece_with_geometry(doc, wp_geo, "wp1")
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        # Get the stock polygons that would be passed to deepnest
        stock_polygons = strategy._get_stock_polygons()

        assert len(stock_polygons) == 1
        sheet_geo, sheet_uid = stock_polygons[0]

        # The sheet geometry should be the L-shape with reasonable size
        sheet_rect = sheet_geo.rect()

        # Sheet geometry should be at world position (50, 50)
        assert sheet_rect[0] == pytest.approx(50)
        assert sheet_rect[1] == pytest.approx(50)

        # The L-shape extends to 200x100 (from (50,50) to (250,150))
        sheet_width = sheet_rect[2] - sheet_rect[0]
        sheet_height = sheet_rect[3] - sheet_rect[1]
        assert sheet_width == pytest.approx(200, rel=0.1)
        assert sheet_height == pytest.approx(100, rel=0.1)

    def test_sheared_stock_sheet_has_correct_size(self):
        """
        Test that sheet polygon for sheared stock is correct.

        BUG: Sheared stock causes incorrect placement.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(100, 0)
        stock_geo.line_to(100, 50)
        stock_geo.line_to(0, 50)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(100, 50)
        stock_matrix = Matrix.compose(50, 50, 0, 1, 1, 30) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(20, 0)
        wp_geo.line_to(20, 20)
        wp_geo.line_to(0, 20)
        wp_geo.close_path()

        wp1 = _create_workpiece_with_geometry(doc, wp_geo, "wp1")
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        # Get the stock polygons that would be passed to deepnest
        stock_polygons = strategy._get_stock_polygons()

        assert len(stock_polygons) == 1
        sheet_geo, sheet_uid = stock_polygons[0]

        # The sheet geometry should be approximately 100x50
        sheet_rect = sheet_geo.rect()
        sheet_width = sheet_rect[2] - sheet_rect[0]
        sheet_height = sheet_rect[3] - sheet_rect[1]

        # Shear may change bbox slightly, but should be close to 100x50
        assert sheet_width > 90 and sheet_width < 150, (
            f"Sheet width {sheet_width} seems incorrect for "
            f"sheared 100x50 stock"
        )
        assert sheet_height > 40 and sheet_height < 100, (
            f"Sheet height {sheet_height} seems incorrect for "
            f"sheared 100x50 stock"
        )

    def test_workpiece_with_no_local_transform_uses_correct_offset(self):
        """
        Test that workpiece with identity local matrix gets correct position.

        BUG: Stock offset (50, 50) is being applied twice when workpiece
        is at origin (identity matrix).
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(200, 100)
        stock_matrix = Matrix.translation(50, 50) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(20, 0)
        wp_geo.line_to(20, 20)
        wp_geo.line_to(0, 20)
        wp_geo.close_path()

        # Create workpiece with identity matrix (at origin)
        source = SourceAsset(
            source_file=Path("wp1.svg"),
            original_data=b"<svg></svg>",
            renderer=SVG_RENDERER,
        )
        doc.add_asset(source)

        pristine_geo = Geometry()
        pristine_geo.move_to(0, 0)
        pristine_geo.line_to(1, 0)
        pristine_geo.line_to(1, 1)
        pristine_geo.line_to(0, 1)
        pristine_geo.close_path()

        norm_matrix = Matrix.scale(20, 20)

        segment = SourceAssetSegment(
            source_asset_uid=source.uid,
            pristine_geometry=pristine_geo,
            normalization_matrix=norm_matrix,
            vectorization_spec=PassthroughSpec(),
        )

        wp1 = WorkPiece(name="wp1", source_segment=segment)
        doc.active_layer.add_child(wp1)

        # Add a second workpiece to meet the minimum requirement of 2
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        # wp1.matrix is identity by default, so it's at world origin (0,0)
        # It should be placed at (50, 50) after nesting (the stock position)
        # NOT at (100, 100) (double offset)

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        deltas = strategy.calculate_deltas()

        if wp1 in deltas:
            delta = deltas[wp1]
            new_world = delta @ wp1.get_world_transform()

            # Check final position
            tx, ty = new_world.get_translation()

            # With stock at (50, 50) and 20x20 workpiece,
            # position should be in range [50, 70], NOT [100, 120]
            assert 40 < tx < 90, (
                f"Workpiece x={tx} suggests double offset (expected 50-70)"
            )
            assert 40 < ty < 90, (
                f"Workpiece y={ty} suggests double offset (expected 50-70)"
            )
        else:
            # If no placement, that's also a bug - 20x20 workpiece
            # should fit in 200x100 stock
            assert False, "Workpiece should have been placed"

    def test_stock_offset_not_applied_twice(self):
        """
        Test that stock offset is not applied twice to workpieces.

        BUG: When stock has translation (50, 50), workpieces get offset by
        (100, 100) instead of just (50, 50), causing them to be placed at
        bottom edge of screen instead of within the stock.

        This happens because:
        1. Stock matrix includes translation
        2. Workpiece world transform includes stock translation (via hierarchy)
        3. Delta matrix is computed incorrectly and adds translation again
        4. Final position = 3x translation instead of 1x
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(200, 100)
        stock_matrix = Matrix.translation(50, 50) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(20, 0)
        wp_geo.line_to(20, 20)
        wp_geo.line_to(0, 20)
        wp_geo.close_path()

        wp1 = _create_workpiece_with_geometry(doc, wp_geo, "wp1")
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        deltas = strategy.calculate_deltas()

        assert len(deltas) >= 1, "Should have at least one placement"

        # Check that the final position accounts for stock offset correctly
        for wp in [wp1, wp2]:
            if wp not in deltas:
                continue

            delta = deltas[wp]
            new_world = delta @ wp.get_world_transform()

            # The workpiece should be placed near (50, 50) (the stock position)
            # NOT near (100, 100) (double the offset)
            translation = new_world.get_translation()

            # Workpiece is 20x20, so should be somewhere in [50, 70] range
            # It should NOT be at 100+ (which would be 2x the offset)
            assert translation[0] < 90, (
                f"Workpiece x position {translation[0]} too high, "
                f"offset likely applied twice. Expected <90, "
                f"got {translation[0]}"
            )
            assert translation[0] > 40, (
                f"Workpiece x position {translation[0]} too low"
            )

    def test_scaled_stock_sheet_polygon_has_correct_size(self):
        """
        Test that sheet polygon passed to deepnest is correctly sized
        when stock is scaled.

        BUG: Scaled stock causes sheet polygon to be wrong size.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(400, 200)
        stock_matrix = Matrix.translation(50, 50) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(20, 0)
        wp_geo.line_to(20, 20)
        wp_geo.line_to(0, 20)
        wp_geo.close_path()

        wp1 = _create_workpiece_with_geometry(doc, wp_geo, "wp1")
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        # Get the stock polygons that would be passed to deepnest
        stock_polygons = strategy._get_stock_polygons()

        assert len(stock_polygons) == 1
        sheet_geo, sheet_uid = stock_polygons[0]

        # The sheet geometry should be 400x200 (200x100 scaled by 2)
        sheet_rect = sheet_geo.rect()
        sheet_width = sheet_rect[2] - sheet_rect[0]
        sheet_height = sheet_rect[3] - sheet_rect[1]

        assert sheet_width == pytest.approx(400, rel=0.1), (
            f"Sheet width should be 400 (200x2), got {sheet_width}"
        )
        assert sheet_height == pytest.approx(200, rel=0.1), (
            f"Sheet height should be 200 (100x2), got {sheet_height}"
        )

    def test_workpiece_placed_within_translated_stock_bounds(self):
        """
        Test that a workpiece is placed within stock bounds
        when stock is translated.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(200, 100)
        stock_matrix = Matrix.translation(50, 50) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(20, 0)
        wp_geo.line_to(20, 20)
        wp_geo.line_to(0, 20)
        wp_geo.close_path()

        wp1 = _create_workpiece_with_geometry(doc, wp_geo, "wp1")
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        deltas = strategy.calculate_deltas()

        assert len(deltas) == 2

        stock_world_geo = stock.get_world_geometry()
        stock_rect = stock_world_geo.rect()

        for wp in [wp1, wp2]:
            delta = deltas[wp]
            new_world = delta @ wp.get_world_transform()

            wp_world_geo = wp.get_world_geometry()
            assert wp_world_geo is not None
            wp_world_geo.transform(new_world.to_4x4_numpy())
            wp_rect = wp_world_geo.rect()

            assert wp_rect[0] >= stock_rect[0]
            assert wp_rect[1] >= stock_rect[1]
            assert wp_rect[2] <= stock_rect[2]
            assert wp_rect[3] <= stock_rect[3]

    def test_workpiece_placed_within_rotated_stock_bounds(self):
        """
        Test that a workpiece is placed within stock bounds
        when stock is rotated.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(200, 100)
        stock_matrix = Matrix.rotation(30) @ stock_matrix
        stock_matrix = Matrix.translation(100, 100) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(20, 0)
        wp_geo.line_to(20, 20)
        wp_geo.line_to(0, 20)
        wp_geo.close_path()

        wp1 = _create_workpiece_with_geometry(doc, wp_geo, "wp1")
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        deltas = strategy.calculate_deltas()

        assert len(deltas) == 2

        stock_world_geo = stock.get_world_geometry()
        stock_rect = stock_world_geo.rect()

        for wp in [wp1, wp2]:
            delta = deltas[wp]
            new_world = delta @ wp.get_world_transform()

            wp_world_geo = wp.get_world_geometry()
            assert wp_world_geo is not None
            wp_world_geo.transform(new_world.to_4x4_numpy())
            wp_rect = wp_world_geo.rect()

            assert wp_rect[0] >= stock_rect[0]
            assert wp_rect[1] >= stock_rect[1]
            assert wp_rect[2] <= stock_rect[2]
            assert wp_rect[3] <= stock_rect[3]

    def test_workpiece_placed_within_scaled_stock_bounds(self):
        """
        Test that a workpiece is placed within stock bounds
        when stock is scaled.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(400, 200)
        stock_matrix = Matrix.translation(50, 50) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(20, 0)
        wp_geo.line_to(20, 20)
        wp_geo.line_to(0, 20)
        wp_geo.close_path()

        wp1 = _create_workpiece_with_geometry(doc, wp_geo, "wp1")
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        deltas = strategy.calculate_deltas()

        assert len(deltas) == 2

        stock_world_geo = stock.get_world_geometry()
        stock_rect = stock_world_geo.rect()

        for wp in [wp1, wp2]:
            delta = deltas[wp]
            new_world = delta @ wp.get_world_transform()

            wp_world_geo = wp.get_world_geometry()
            assert wp_world_geo is not None
            wp_world_geo.transform(new_world.to_4x4_numpy())
            wp_rect = wp_world_geo.rect()

            assert wp_rect[0] >= stock_rect[0]
            assert wp_rect[1] >= stock_rect[1]
            assert wp_rect[2] <= stock_rect[2]
            assert wp_rect[3] <= stock_rect[3]

    def test_non_rectangular_stock_contains_workpiece(self):
        """
        Test that workpiece is placed within the actual L-shaped stock,
        not just its bounding box.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 25)
        stock_geo.line_to(25, 25)
        stock_geo.line_to(25, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(200, 100)
        stock_matrix = Matrix.translation(50, 50) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(15, 0)
        wp_geo.line_to(15, 15)
        wp_geo.line_to(0, 15)
        wp_geo.close_path()

        wp1 = _create_workpiece_with_geometry(doc, wp_geo, "wp1")
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        deltas = strategy.calculate_deltas()

        assert len(deltas) >= 1

        for wp in [wp1, wp2]:
            if wp not in deltas:
                continue
            delta = deltas[wp]
            new_world = delta @ wp.get_world_transform()

            wp_world_geo = wp.get_world_geometry()
            assert wp_world_geo is not None
            wp_world_geo.transform(new_world.to_4x4_numpy())
            wp_rect = wp_world_geo.rect()

            assert wp_rect[0] >= 50
            assert wp_rect[1] >= 50
            assert wp_rect[2] <= 250
            assert wp_rect[3] <= 150

    def test_sheared_stock_contains_workpiece(self):
        """
        Test that workpiece is placed within sheared stock bounds.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(100, 0)
        stock_geo.line_to(100, 50)
        stock_geo.line_to(0, 50)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(100, 50)
        stock_matrix = Matrix.compose(50, 50, 0, 1, 1, 30) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(20, 0)
        wp_geo.line_to(20, 20)
        wp_geo.line_to(0, 20)
        wp_geo.close_path()

        wp1 = _create_workpiece_with_geometry(doc, wp_geo, "wp1")
        wp2 = _create_workpiece_with_geometry(doc, wp_geo, "wp2")

        strategy = NestingLayoutStrategy([stock, wp1, wp2])
        strategy.rotations = 4
        strategy.population_size = 4

        deltas = strategy.calculate_deltas()

        assert len(deltas) >= 1

        for wp in [wp1, wp2]:
            if wp not in deltas:
                continue
            delta = deltas[wp]
            new_world = delta @ wp.get_world_transform()

            wp_world_geo = wp.get_world_geometry()
            assert wp_world_geo is not None
            wp_world_geo.transform(new_world.to_4x4_numpy())
            wp_rect = wp_world_geo.rect()

            stock_world_geo = stock.get_world_geometry()
            stock_rect = stock_world_geo.rect()

            assert wp_rect[0] >= stock_rect[0]
            assert wp_rect[1] >= stock_rect[1]
            assert wp_rect[2] <= stock_rect[2]
            assert wp_rect[3] <= stock_rect[3]


class TestPlacementDeltaComputation:
    """Tests for how deltas are computed from placements."""

    def test_delta_accounts_for_stock_offset(self):
        """
        Test that when a stock is at position (100, 50), the workpiece
        placement delta correctly accounts for this offset.

        Test verifies that workpieces can be placed in stock.
        """
        stock_geo = Geometry()
        stock_geo.move_to(0, 0)
        stock_geo.line_to(200, 0)
        stock_geo.line_to(200, 100)
        stock_geo.line_to(0, 100)
        stock_geo.close_path()

        stock_matrix = Matrix.scale(200, 100)
        stock_matrix = Matrix.translation(100, 50) @ stock_matrix

        doc = Doc()
        stock = _create_stock_with_geometry(doc, stock_geo, stock_matrix)

        wp_geo = Geometry()
        wp_geo.move_to(0, 0)
        wp_geo.line_to(30, 0)
        wp_geo.line_to(30, 30)
        wp_geo.line_to(0, 30)
        wp_geo.close_path()

        wp = _create_workpiece_with_geometry(doc, wp_geo, "wp1")

        strategy = NestingLayoutStrategy([stock, wp])
        strategy.rotations = 4
        strategy.population_size = 4

        # Placement.x and Placement.y are world space coordinates
        # (including sheet offset), not relative to normalized sheet
        placement = Placement(
            id=0,
            source=0,
            uid=wp.uid,
            x=110,
            y=60,
            rotation=0,
            polygons=[],
            sheet_uid="stock-0",
        )

        deltas = strategy._compute_deltas_from_placements([placement], [wp])

        assert len(deltas) == 1
        delta = deltas[wp]

        new_world = delta @ wp.get_world_transform()
        new_pos = new_world.transform_point((0, 0))

        # Workpiece should be placed at the placement position (110, 60)
        assert new_pos[0] == pytest.approx(110)
        assert new_pos[1] == pytest.approx(60)
