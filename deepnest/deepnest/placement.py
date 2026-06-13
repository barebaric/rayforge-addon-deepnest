import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from raygeo.geo.shape.polygon import (
    polygon_area_numpy,
    polygon_bounds_numpy,
    point_in_polygon_numpy,
    translate_polygons_numpy,
)
from raygeo.nest import collision as _collision
from raygeo.nest import placement as _placement
from .models import NestConfig, Placement, SheetInfo

logger = logging.getLogger(__name__)

NumpyPolygon = np.ndarray


@dataclass
class NestResult:
    placements: List[Placement]
    fitness: float
    area_used: float
    sheet_index: int = 0


def layout_sheets_horizontal(
    sheets: List[SheetInfo],
    spacing: float = 0.0,
) -> List[SheetInfo]:
    """Arrange sheets horizontally in unified world space.

    If sheets already have non-zero offsets (from document positions),
    they are used as-is. Otherwise, they are arranged with horizontal spacing.
    """
    has_existing_offsets = any(
        s.world_offset_x != 0 or s.world_offset_y != 0 for s in sheets
    )
    if has_existing_offsets:
        return sheets

    x_offset = 0.0
    for sheet in sheets:
        bounds = polygon_bounds_numpy(sheet.polygon)
        width = bounds[2] - bounds[0]
        sheet.world_offset_x = x_offset
        x_offset += width + spacing
    return sheets


def get_sheet_at_position(
    x: float,
    y: float,
    sheets: List[SheetInfo],
) -> Optional[SheetInfo]:
    """Find which sheet contains the given position."""
    for sheet in sheets:
        poly_at_offset = translate_polygons_numpy(
            [sheet.polygon], sheet.world_offset_x, sheet.world_offset_y
        )[0]
        if point_in_polygon_numpy((x, y), poly_at_offset):
            return sheet
    return None


def place_parts(
    parts: List[Dict[str, Any]],
    sheets: List[SheetInfo],
    rotations: List[float],
    config: NestConfig,
    sheet_spacing: float = 0.0,
    flips_h: Optional[List[bool]] = None,
    flips_v: Optional[List[bool]] = None,
) -> Optional[NestResult]:
    """
    Place parts using the Rust nesting engine.

    Parts are sorted by area (largest first) for better performance.
    Supports multi-sheet nesting via unified world space.
    """
    if not parts or not sheets:
        logger.warning("place_parts: no parts or sheets")
        return None

    num_parts = len(parts)
    layout_sheets_horizontal(sheets, sheet_spacing)

    if not sheets:
        logger.warning("place_parts: no sheets after layout")
        return None

    logger.info(
        "place_parts: starting with %d parts, %d sheets",
        num_parts,
        len(sheets),
    )
    for sheet in sheets:
        sheet_bounds = polygon_bounds_numpy(sheet.polygon)
        logger.info(
            "  Sheet '%s': bounds=(%.2f,%.2f)-(%.2f,%.2f), offset=(%.2f,%.2f)",
            sheet.uid,
            sheet_bounds[0],
            sheet_bounds[1],
            sheet_bounds[2],
            sheet_bounds[3],
            sheet.world_offset_x,
            sheet.world_offset_y,
        )

    # Convert parts to Rust format: list of (polygons, hulls)
    part_polys = []
    part_hulls = []
    for part in parts:
        polys = part.get("polygons", [])
        hulls = part.get("hulls", [])
        part_polys.append(
            [[(float(vx), float(vy)) for vx, vy in poly] for poly in polys]
        )
        part_hulls.append(
            [[(float(vx), float(vy)) for vx, vy in hull] for hull in hulls]
        )

    # Convert sheets to Rust format
    sheet_polys = [
        [(float(vx), float(vy)) for vx, vy in sheet.polygon]
        for sheet in sheets
    ]
    sheet_offsets = [
        (sheet.world_offset_x, sheet.world_offset_y) for sheet in sheets
    ]

    rotations_deg = list(rotations)
    flips_h_list = list(flips_h) if flips_h else [False] * num_parts
    flips_v_list = list(flips_v) if flips_v else [False] * num_parts

    spacing = config.spacing
    curve_tolerance = config.curve_tolerance

    rust_results = _placement.place_parts(
        part_polys,
        part_hulls,
        sheet_polys,
        sheet_offsets,
        rotations_deg,
        flips_h_list,
        flips_v_list,
        spacing=spacing,
        min_area=10.0,
        curve_tolerance=curve_tolerance,
    )

    if not rust_results:
        logger.warning("place_parts: no placements made")
        return None

    # Collect all placements across sheets and convert to Python objects
    sheet_uid_map = {i: sheet.uid for i, sheet in enumerate(sheets)}
    all_placements: List[Placement] = []

    for sheet_result in rust_results:
        sheet_uid = sheet_uid_map.get(sheet_result["sheet_index"], "unknown")
        for pl in sheet_result["placements"]:
            part_index = pl["part_index"]
            part = parts[part_index]

            all_placements.append(
                Placement(
                    id=part.get("id", part_index),
                    source=part.get("source", part_index),
                    uid=part.get("uid", f"part_{part_index}"),
                    x=pl["position"][0],
                    y=pl["position"][1],
                    rotation=rotations_deg[part_index],
                    polygons=[
                        np.array(poly, dtype=np.float64)
                        for poly in pl["polygons"]
                    ],
                    hulls=[
                        np.array(hull, dtype=np.float64)
                        for hull in pl["hulls"]
                    ],
                    sheet_uid=sheet_uid,
                    flip_h=flips_h_list[part_index],
                    flip_v=flips_v_list[part_index],
                )
            )

    if not all_placements:
        logger.warning("place_parts: no placements made")
        return None

    fitness = rust_results[0]["fitness"]

    total_area = sum(
        abs(polygon_area_numpy(poly))
        for p in all_placements
        for poly in p.polygons
    )

    logger.debug(
        "Placement result: %d/%d placements, fitness=%.4f",
        len(all_placements),
        num_parts,
        fitness,
    )

    return NestResult(
        placements=all_placements,
        fitness=fitness,
        area_used=total_area,
        sheet_index=0,
    )


def validate_placements_no_overlap(
    placements: List[Placement],
    min_overlap_area: float = 100,
) -> bool:
    """
    Validate that placements don't overlap after transformation.

    Returns True if valid (no overlaps), False if overlaps detected.
    Logs warnings for any overlaps found.
    """
    if len(placements) < 2:
        return True
    for i, p1 in enumerate(placements):
        for j, p2 in enumerate(placements):
            if i >= j:
                continue
            for poly1 in p1.polygons:
                for poly2 in p2.polygons:
                    if _collision.any_overlap(
                        poly1.tolist(), [poly2.tolist()], min_overlap_area
                    ):
                        return False
    return True
