"""
Tests for multi-sheet nesting functionality.
"""

import pytest
import numpy as np

from deepnest.deepnest.models import (
    NestConfig,
    SheetInfo,
)
from deepnest.deepnest.core import DeepNest
from deepnest.deepnest.placement import (
    place_parts,
    layout_sheets_horizontal,
    get_sheet_at_position,
)


def P(*points):
    """Helper to create a polygon from integer points."""
    return np.array([(float(x), float(y)) for x, y in points])


def SI(polygon: np.ndarray, uid="sheet") -> SheetInfo:
    """Helper to create a SheetInfo from a polygon."""
    return SheetInfo(uid=uid, polygon=polygon)


class TestAddSheetMethod:
    """Tests for DeepNest.add_sheet() method."""

    @pytest.fixture
    def nester(self):
        return DeepNest(NestConfig())

    def test_add_sheet_basic(self, nester):
        """add_sheet should add a sheet polygon."""
        sheet_poly = P((0, 0), (100, 0), (100, 100), (0, 100))
        result = nester.add_sheet(sheet_poly, uid="test-sheet")
        assert result is True
        assert len(nester._sheets) == 1
        assert nester._sheets[0].uid == "test-sheet"
        assert nester._sheets[0].is_sheet is True

    def test_add_sheet_auto_uid(self, nester):
        """add_sheet should generate uid if not provided."""
        sheet_poly = P((0, 0), (50, 0), (50, 50), (0, 50))
        result = nester.add_sheet(sheet_poly)
        assert result is True
        assert len(nester._sheets) == 1
        assert nester._sheets[0].uid.startswith("sheet_")

    def test_add_multiple_sheets(self, nester):
        """Multiple sheets should be stored separately."""
        sheet1 = P((0, 0), (100, 0), (100, 100), (0, 100))
        sheet2 = P((0, 0), (50, 0), (50, 50), (0, 50))
        nester.add_sheet(sheet1, uid="sheet-1")
        nester.add_sheet(sheet2, uid="sheet-2")
        assert len(nester._sheets) == 2
        assert nester._sheets[0].uid == "sheet-1"
        assert nester._sheets[1].uid == "sheet-2"


class TestLayoutSheetsHorizontal:
    """Tests for layout_sheets_horizontal function."""

    def test_single_sheet_no_offset(self):
        """Single sheet should have zero offset."""
        sheet_poly = P((0, 0), (100, 0), (100, 100), (0, 100))
        sheets = [SI(sheet_poly)]
        result = layout_sheets_horizontal(sheets)
        assert len(result) == 1
        assert result[0].world_offset_x == 0.0
        assert result[0].world_offset_y == 0.0

    def test_two_sheets_horizontal_layout(self):
        """Two sheets should be laid out horizontally."""
        sheet0_poly = P((0, 0), (100, 0), (100, 100), (0, 100))
        sheet1_poly = P((0, 0), (50, 0), (50, 50), (0, 50))
        sheets = [
            SI(sheet0_poly),
            SI(sheet1_poly),
        ]
        result = layout_sheets_horizontal(sheets)
        assert len(result) == 2
        assert result[0].world_offset_x == 0.0
        assert result[1].world_offset_x == 100.0

    def test_three_sheets_with_spacing(self):
        """Three sheets with spacing should have gaps."""
        sheet_poly = P((0, 0), (50, 0), (50, 50), (0, 50))
        sheets = [
            SI(sheet_poly),
            SI(sheet_poly),
            SI(sheet_poly),
        ]
        result = layout_sheets_horizontal(sheets, spacing=10)
        assert len(result) == 3
        assert result[0].world_offset_x == 0.0
        assert result[1].world_offset_x == 60.0
        assert result[2].world_offset_x == 120.0

    def test_sheets_with_existing_offsets_preserved(self):
        """Sheets with existing offsets should be preserved as-is."""
        sheet_poly = P((0, 0), (100, 0), (100, 100), (0, 100))
        sheet0 = SI(sheet_poly, "sheet-0")
        sheet0.world_offset_x = 0.0
        sheet0.world_offset_y = 0.0
        sheet1 = SI(sheet_poly, "sheet-1")
        sheet1.world_offset_x = 150.0
        sheet1.world_offset_y = 0.0
        sheets = [sheet0, sheet1]

        result = layout_sheets_horizontal(sheets)

        assert len(result) == 2
        assert result[0].world_offset_x == 0.0
        assert result[1].world_offset_x == 150.0

    def test_sheets_with_any_nonzero_offset_skips_layout(self):
        """If any sheet has non-zero offset, layout is skipped for all."""
        sheet_poly = P((0, 0), (100, 0), (100, 100), (0, 100))
        sheet0 = SI(sheet_poly, "sheet-0")
        sheet0.world_offset_x = 0.0
        sheet1 = SI(sheet_poly, "sheet-1")
        sheet1.world_offset_x = 200.0
        sheet2 = SI(sheet_poly, "sheet-2")
        sheet2.world_offset_x = 400.0
        sheets = [sheet0, sheet1, sheet2]

        result = layout_sheets_horizontal(sheets)

        assert len(result) == 3
        assert result[0].world_offset_x == 0.0
        assert result[1].world_offset_x == 200.0
        assert result[2].world_offset_x == 400.0


class TestGetSheetAtPosition:
    """Tests for get_sheet_at_position function."""

    def test_position_in_first_sheet(self):
        """Position in first sheet should return first sheet."""
        sheet_poly = P((0, 0), (100, 0), (100, 100), (0, 100))
        sheets = [
            SI(sheet_poly),
            SI(sheet_poly),
        ]
        layout_sheets_horizontal(sheets)
        result = get_sheet_at_position(50, 50, sheets)
        assert result is not None
        assert result.uid == "sheet"

    def test_position_in_second_sheet(self):
        """Position in second sheet should return second sheet."""
        sheet_poly = P((0, 0), (100, 0), (100, 100), (0, 100))
        sheets = [
            SI(sheet_poly),
            SI(sheet_poly),
        ]
        layout_sheets_horizontal(sheets)
        result = get_sheet_at_position(150, 50, sheets)
        assert result is not None
        assert result.uid == "sheet"

    def test_position_outside_all_sheets(self):
        """Position outside all sheets should return None."""
        sheet_poly = P((0, 0), (100, 0), (100, 100), (0, 100))
        sheets = [SI(sheet_poly)]
        layout_sheets_horizontal(sheets)
        result = get_sheet_at_position(200, 200, sheets)
        assert result is None


class TestSingleSheetBackwardCompat:
    """Tests for backward compatibility with single sheet."""

    @pytest.fixture
    def config(self):
        return NestConfig(curve_tolerance=0.5)

    def test_single_sheet_works_like_before(self, config):
        """Single sheet should work the same as before."""
        parts = [
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
        sheets = [SI(P((0, 0), (100, 0), (100, 100), (0, 100)))]
        rotations = [0.0, 0.0]
        result = place_parts(parts, sheets, rotations, config)
        assert result is not None
        assert len(result.placements) == 2


class TestMultiSheetPlacement:
    """Tests for placing parts on multiple sheets."""

    @pytest.fixture
    def config(self):
        return NestConfig(curve_tolerance=0.5)

    def test_parts_on_multiple_sheets(self, config):
        """Parts should be placed across multiple sheets."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"part-{i}",
                "polygons": [P((0, 0), (60, 0), (60, 60), (0, 60))],
            }
            for i in range(3)
        ]
        sheets = [
            SI(P((0, 0), (100, 0), (100, 100), (0, 100)), "sheet-0"),
            SI(P((0, 0), (100, 0), (100, 100), (0, 100)), "sheet-1"),
            SI(P((0, 0), (100, 0), (100, 100), (0, 100)), "sheet-2"),
        ]
        rotations = [0.0, 0.0, 0.0]
        result = place_parts(parts, sheets, rotations, config)
        assert result is not None
        assert len(result.placements) == 3

    def test_sheet_uid_in_placements(self, config):
        """Each placement should have correct sheet_uid."""
        parts = [
            {
                "id": i,
                "source": i,
                "uid": f"part-{i}",
                "polygons": [P((0, 0), (40, 0), (40, 40), (0, 40))],
            }
            for i in range(4)
        ]
        sheets = [
            SI(P((0, 0), (100, 0), (100, 100), (0, 100)), "sheet-0"),
            SI(P((0, 0), (100, 0), (100, 100), (0, 100)), "sheet-1"),
        ]
        rotations = [0.0] * 4
        result = place_parts(parts, sheets, rotations, config)
        assert result is not None
        for placement in result.placements:
            assert placement.sheet_uid is not None
            assert placement.sheet_uid in ["sheet-0", "sheet-1"]
