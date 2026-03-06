import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pyclipper
import math

from rayforge.core.geo import Rect
from rayforge.core.geo.polygon import (
    Polygon,
    point_in_polygon,
    polygon_area,
    polygon_bounds,
    polygon_group_bounds,
    polygons_intersect,
    rotate_polygons,
    translate_polygons,
    to_clipper,
    from_clipper,
    normalize_polygons,
    polygon_offset,
)
from rayforge.core.geo.query import bboxes_intersect
from .models import NestConfig, Placement, SheetInfo
from .nfp import inner_fit_polygon, no_fit_polygon
from .spatial_grid import SpatialGrid

logger = logging.getLogger(__name__)


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
    # If any sheet has existing offsets from document positions,
    # assume all sheets are already positioned correctly.
    has_existing_offsets = any(
        s.world_offset_x != 0 or s.world_offset_y != 0 for s in sheets
    )
    if has_existing_offsets:
        return sheets

    x_offset = 0.0
    for sheet in sheets:
        bounds = polygon_bounds(sheet.polygon)
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
        poly_at_offset = translate_polygons(
            [sheet.polygon], sheet.world_offset_x, sheet.world_offset_y
        )[0]
        if point_in_polygon((x, y), poly_at_offset):
            return sheet
    return None


def _any_overlap(
    candidate_polys: List[Polygon],
    placed_polys_list: List[List[Polygon]],
    spatial_grid: Optional[SpatialGrid] = None,
    candidate_bbox: Optional[Rect] = None,
) -> bool:
    """Check if candidate polygons overlap with any placed polygons."""
    cand_bbox = candidate_bbox or polygon_group_bounds(candidate_polys)

    if spatial_grid is not None:
        nearby_indices = spatial_grid.query(cand_bbox)
        for idx in nearby_indices:
            if idx >= len(placed_polys_list):
                continue
            placed_polys = placed_polys_list[idx]
            placed_bbox = polygon_group_bounds(placed_polys)
            if not bboxes_intersect(cand_bbox, placed_bbox):
                continue
            for cand_poly in candidate_polys:
                for placed_poly in placed_polys:
                    if polygons_intersect(cand_poly, placed_poly, min_area=10):
                        return True
    else:
        for placed_polys in placed_polys_list:
            placed_bbox = polygon_group_bounds(placed_polys)
            if not bboxes_intersect(cand_bbox, placed_bbox):
                continue
            for cand_poly in candidate_polys:
                for placed_poly in placed_polys:
                    if polygons_intersect(cand_poly, placed_poly, min_area=10):
                        return True
    return False


def _is_contained(
    inner_polys: List[Polygon], outer_poly: Polygon, scale: int
) -> bool:
    """
    Check if inner_polys are strictly contained within outer_poly using
    Clipper difference.
    If (Inner - Outer) is not empty, then Inner is sticking out.
    """
    try:
        clipper = pyclipper.Pyclipper()
        # Add outer as CLIP (to subtract it)
        clipper.AddPath(to_clipper(outer_poly, scale), pyclipper.PT_CLIP, True)

        # Add inners as SUBJECT
        for p in inner_polys:
            clipper.AddPath(to_clipper(p, scale), pyclipper.PT_SUBJECT, True)

        # Execute Difference: Subject - Clip
        result = clipper.Execute(
            pyclipper.CT_DIFFERENCE,
            pyclipper.PFT_NONZERO,
            pyclipper.PFT_NONZERO,
        )

        if not result:
            return True

        # Allow for negligible floating point noise
        total_area = sum(abs(pyclipper.Area(path)) for path in result)
        return total_area < 100

    except Exception:
        return False


def _get_combined_ifp(
    sheet: Polygon,
    part_polygons: List[Polygon],
    config: NestConfig,
) -> List[Polygon]:
    """Get the combined IFP for all polygons in a part using intersection."""
    if not part_polygons:
        return []

    combined_ifps = []
    scale = config.clipper_scale

    for poly in part_polygons:
        ifps = inner_fit_polygon(sheet, poly, config)
        if not ifps:
            return []

        if not combined_ifps:
            combined_ifps = ifps
        else:
            try:
                clipper = pyclipper.Pyclipper()
                for c_ifp in combined_ifps:
                    clipper.AddPath(
                        to_clipper(c_ifp, scale),
                        pyclipper.PT_SUBJECT,
                        True,
                    )
                for ifp in ifps:
                    clipper.AddPath(
                        to_clipper(ifp, scale),
                        pyclipper.PT_CLIP,
                        True,
                    )
                result = clipper.Execute(
                    pyclipper.CT_INTERSECTION,
                    pyclipper.PFT_NONZERO,
                    pyclipper.PFT_NONZERO,
                )
                if result:
                    combined_ifps = [from_clipper(p, scale) for p in result]
                else:
                    return []
            except Exception:
                return []

    return combined_ifps


def _generate_perimeter_candidates(
    placed_polys_list: List[List[Polygon]],
    part_bounds: Tuple[float, float, float, float],
    spacing: float,
    ifp_bounds: Tuple[float, float, float, float],
    max_candidates: int = 500,
) -> List[Tuple[float, float]]:
    """
    Generate candidate positions along the perimeter of already-placed parts.
    Uses bounding box corners and edge midpoints for fast heuristic placement.
    """
    candidates = []

    for placed_polys in placed_polys_list:
        p_bounds = polygon_group_bounds(placed_polys)

        left_x = p_bounds[0] - part_bounds[2] - spacing
        right_x = p_bounds[2] - part_bounds[0] + spacing
        bottom_y = p_bounds[1] - part_bounds[3] - spacing
        top_y = p_bounds[3] - part_bounds[1] + spacing

        candidates.extend(
            [
                (left_x, bottom_y),
                (right_x, bottom_y),
                (right_x, top_y),
                (left_x, top_y),
            ]
        )

        for placed_poly in placed_polys:
            for px, py in placed_poly:
                cand_x = px - part_bounds[2] - spacing
                cand_y = py - part_bounds[3] - spacing
                candidates.append((cand_x, cand_y))
                candidates.append((px - part_bounds[0] + spacing, cand_y))
                candidates.append((cand_x, py - part_bounds[1] + spacing))
                candidates.append(
                    (
                        px - part_bounds[0] + spacing,
                        py - part_bounds[1] + spacing,
                    )
                )

    return candidates[:max_candidates]


def _generate_bottom_left_candidates(
    ifp_bounds: Tuple[float, float, float, float],
    part_bounds: Tuple[float, float, float, float],
    spacing: float,
    step_size: float = 10.0,
) -> List[Tuple[float, float]]:
    """
    Generate candidates along the bottom and left edges of the IFP.
    This implements the classic bottom-left placement heuristic.
    """
    candidates = []

    min_x = ifp_bounds[0] - part_bounds[0] + spacing
    max_x = ifp_bounds[2] - part_bounds[2] - spacing
    min_y = ifp_bounds[1] - part_bounds[1] + spacing
    max_y = ifp_bounds[3] - part_bounds[3] - spacing

    candidates.append((min_x, min_y))
    candidates.append((min_x, min_y + step_size))
    candidates.append((min_x + step_size, min_y))

    x = min_x
    while x <= max_x:
        candidates.append((x, min_y))
        x += step_size

    y = min_y
    while y <= max_y:
        candidates.append((min_x, y))
        y += step_size

    return candidates


def _find_valid_position_fast(
    ifp: Polygon,
    part_polygons: List[Polygon],
    placed_polys_list: List[List[Polygon]],
    config: NestConfig,
    spacing: float = 0.1,
    spatial_grid: Optional[SpatialGrid] = None,
    sheet_world_offset: Tuple[float, float] = (0.0, 0.0),
) -> Optional[Tuple[float, float]]:
    """
    Fast placement using bottom-left and perimeter-based heuristics.
    Avoids expensive NFP subtraction by testing candidate positions directly.
    """
    if not ifp or len(ifp) < 3:
        return None

    offset_x, offset_y = sheet_world_offset
    ifp_world = translate_polygons([ifp], offset_x, offset_y)[0]

    ifp_bounds = polygon_bounds(ifp_world)
    part_bounds = polygon_group_bounds(part_polygons)

    pw = part_bounds[2] - part_bounds[0]
    ph = part_bounds[3] - part_bounds[1]

    candidates = []

    candidates.extend(ifp_world)

    candidates.extend(
        _generate_bottom_left_candidates(ifp_bounds, part_bounds, spacing)
    )

    if placed_polys_list:
        if spatial_grid is not None:
            cand_bbox = (
                ifp_bounds[0] - pw - spacing,
                ifp_bounds[1] - ph - spacing,
                ifp_bounds[2] + pw + spacing,
                ifp_bounds[3] + ph + spacing,
            )
            nearby_indices = spatial_grid.query(cand_bbox)
            parts_to_sample = [
                placed_polys_list[i]
                for i in nearby_indices
                if i < len(placed_polys_list)
            ]
        else:
            parts_to_sample = placed_polys_list

        candidates.extend(
            _generate_perimeter_candidates(
                parts_to_sample, part_bounds, spacing, ifp_bounds
            )
        )

    unique_candidates = set()
    for cx, cy in candidates:
        if cx < ifp_bounds[0] - 0.01 or cx > ifp_bounds[2] + 0.01:
            continue
        if cy < ifp_bounds[1] - 0.01 or cy > ifp_bounds[3] + 0.01:
            continue
        unique_candidates.add((round(cx, 4), round(cy, 4)))

    def score_candidate(pos: Tuple[float, float]) -> float:
        if config.placement_type == "gravity":
            return pos[1] + pos[0] * 0.001
        else:
            return pos[0] + pos[1] * 0.001

    sorted_candidates = sorted(unique_candidates, key=score_candidate)

    best_score = float("inf")
    best_pos = None

    for x, y in sorted_candidates:
        if not point_in_polygon((x, y), ifp_world):
            continue

        score = score_candidate((x, y))
        if score >= best_score:
            continue

        test_polys = translate_polygons(part_polygons, x, y)
        cand_bbox_test = (
            x + part_bounds[0],
            y + part_bounds[1],
            x + part_bounds[2],
            y + part_bounds[3],
        )

        if _any_overlap(
            test_polys, placed_polys_list, spatial_grid, cand_bbox_test
        ):
            continue

        best_score = score
        best_pos = (x, y)
        break

    return best_pos


def _find_valid_position(
    ifp: Polygon,
    part_polygons: List[Polygon],
    placed_polys_list: List[List[Polygon]],
    config: NestConfig,
    spacing: float = 0.1,
    spatial_grid: Optional[SpatialGrid] = None,
    sheet_world_offset: Tuple[float, float] = (0.0, 0.0),
) -> Optional[Tuple[float, float]]:
    """
    Find a valid position in the IFP where part doesn't overlap placed parts.
    Uses fast bottom-left/perimeter heuristic, falls back to NFP subtraction.
    """
    result = _find_valid_position_fast(
        ifp,
        part_polygons,
        placed_polys_list,
        config,
        spacing,
        spatial_grid,
        sheet_world_offset,
    )

    if result is not None:
        return result

    if not ifp or len(ifp) < 3:
        return None

    offset_x, offset_y = sheet_world_offset
    ifp_world = translate_polygons([ifp], offset_x, offset_y)[0]

    ifp_bounds = polygon_bounds(ifp_world)
    part_bounds = polygon_group_bounds(part_polygons)

    # Dimensions of the part
    pw = part_bounds[2] - part_bounds[0]
    ph = part_bounds[3] - part_bounds[1]

    candidates = []

    # Get local placed parts from the spatial grid
    if spatial_grid is not None and placed_polys_list:
        cand_bbox = (
            ifp_bounds[0] - pw - spacing,
            ifp_bounds[1] - ph - spacing,
            ifp_bounds[2] + pw + spacing,
            ifp_bounds[3] + ph + spacing,
        )
        nearby_indices = spatial_grid.query(cand_bbox)
        parts_to_sample = [
            placed_polys_list[i]
            for i in nearby_indices
            if i < len(placed_polys_list)
        ]
    else:
        parts_to_sample = placed_polys_list

    scale = config.clipper_scale
    clipper = pyclipper.Pyclipper()
    clipper.AddPath(to_clipper(ifp_world, scale), pyclipper.PT_SUBJECT, True)

    has_clips = False

    # 1. Gather NFPs from all placed items to subtract from IFP
    for placed_polys in parts_to_sample:
        p_bounds = polygon_group_bounds(placed_polys)

        # 1a. Fallback Bounding Box NFP Candidates
        # Extremely fast and highly optimal corners for axis-aligned
        # nesting behavior
        left_x = p_bounds[0] - part_bounds[2] - spacing
        right_x = p_bounds[2] - part_bounds[0] + spacing
        bottom_y = p_bounds[1] - part_bounds[3] - spacing
        top_y = p_bounds[3] - part_bounds[1] + spacing

        candidates.extend(
            [
                (left_x, bottom_y),
                (right_x, bottom_y),
                (right_x, top_y),
                (left_x, top_y),
            ]
        )

        # 1b. Exact NFP Subtraction
        for placed_poly in placed_polys:
            placed_bbox = polygon_bounds(placed_poly)
            for part_poly in part_polygons:
                part_bbox = polygon_bounds(part_poly)
                pw = part_bbox[2] - part_bbox[0]
                ph = part_bbox[3] - part_bbox[1]

                expanded_placed = (
                    placed_bbox[0] - pw - spacing,
                    placed_bbox[1] - ph - spacing,
                    placed_bbox[2] + pw + spacing,
                    placed_bbox[3] + ph + spacing,
                )

                if not bboxes_intersect(expanded_placed, ifp_bounds):
                    continue

                nfps = no_fit_polygon(placed_poly, part_poly, False, config)
                for nfp in nfps:
                    # Shift NFP backward by the local part_poly's origin
                    # so the region corresponds to the part's (0,0)
                    # placement locus.
                    ox, oy = part_poly[0]
                    nfp_origin = [(px - ox, py - oy) for px, py in nfp]

                    if spacing > 0:
                        expanded = polygon_offset(nfp_origin, spacing)
                        for exp_nfp in expanded:
                            try:
                                clipper.AddPath(
                                    to_clipper(exp_nfp, scale),
                                    pyclipper.PT_CLIP,
                                    True,
                                )
                                has_clips = True
                            except pyclipper.ClipperException:
                                pass
                    else:
                        try:
                            clipper.AddPath(
                                to_clipper(nfp_origin, scale),
                                pyclipper.PT_CLIP,
                                True,
                            )
                            has_clips = True
                        except pyclipper.ClipperException:
                            pass

    # 2. Compute true valid placement regions
    valid_regions = []
    if has_clips:
        try:
            solution = clipper.Execute(
                pyclipper.CT_DIFFERENCE,
                pyclipper.PFT_NONZERO,
                pyclipper.PFT_NONZERO,
            )
            for path in solution:
                valid_regions.append(from_clipper(path, scale))
        except Exception as e:
            logger.warning(
                "Clipper difference failed in NFP valid region generation: %s",
                e,
            )
            valid_regions = [ifp_world]
    else:
        valid_regions = [ifp_world]

    # 3. Harvest candidates exclusively from vertices (Optimal points)
    for region in valid_regions:
        candidates.extend(region)

    # Make sure IFP corner vertices are included in case NFP boolean failed
    candidates.extend(ifp_world)

    # Deduplicate candidate points to avoid redundant intersection checks
    unique_candidates = set()
    for cx, cy in candidates:
        if cx < ifp_bounds[0] - 0.01 or cx > ifp_bounds[2] + 0.01:
            continue
        if cy < ifp_bounds[1] - 0.01 or cy > ifp_bounds[3] + 0.01:
            continue
        unique_candidates.add((round(cx, 4), round(cy, 4)))

    best_score = float("inf")
    best_pos = None

    # 4. Evaluate only the highly-probable candidate points
    for x, y in unique_candidates:
        # Guarantee point is inside the IFP strictly
        if not point_in_polygon((x, y), ifp_world):
            continue

        if config.placement_type == "gravity":
            score = y + x * 0.001
        else:
            score = x + y * 0.001

        if score >= best_score:
            continue

        test_polys = translate_polygons(part_polygons, x, y)
        cand_bbox_test = (
            x + part_bounds[0],
            y + part_bounds[1],
            x + part_bounds[2],
            y + part_bounds[3],
        )

        # Fallback exact collision check to catch any Concave
        # Minkowski artifacts
        if _any_overlap(
            test_polys, placed_polys_list, spatial_grid, cand_bbox_test
        ):
            continue

        best_score = score
        best_pos = (x, y)

    return best_pos


def _apply_gravity(
    placements: List[Placement],
    sheet_poly: Polygon,
    spacing: float,
    clipper_scale: int,
) -> List[Placement]:
    """
    Post-process placements to slide parts down and left as far as possible.
    This removes unnecessary whitespace by applying a gravity effect.
    Iterates Y and X sliding until no more movement occurs.
    """
    if len(placements) < 2:
        return placements

    sheet_bounds = polygon_bounds(sheet_poly)

    for _ in range(10):
        any_moved = False

        sorted_by_y = sorted(
            enumerate(placements),
            key=lambda x: polygon_group_bounds(x[1].polygons)[1],
        )

        for i, placement in sorted_by_y:
            other_polys = [
                p.polygons for j, p in enumerate(placements) if j != i
            ]

            dy = _find_max_slide(
                placement.polygons,
                other_polys,
                sheet_bounds,
                sheet_poly,
                "y",
                spacing,
                clipper_scale,
            )
            if dy > 0.01:
                placement.y -= dy
                placement.polygons = translate_polygons(
                    placement.polygons, 0, -dy
                )
                any_moved = True

        sorted_by_x = sorted(
            enumerate(placements),
            key=lambda x: polygon_group_bounds(x[1].polygons)[0],
        )

        for i, placement in sorted_by_x:
            other_polys = [
                p.polygons for j, p in enumerate(placements) if j != i
            ]

            dx = _find_max_slide(
                placement.polygons,
                other_polys,
                sheet_bounds,
                sheet_poly,
                "x",
                spacing,
                clipper_scale,
            )
            if dx > 0.01:
                placement.x -= dx
                placement.polygons = translate_polygons(
                    placement.polygons, -dx, 0
                )
                any_moved = True

        if not any_moved:
            break

    return placements


def _find_max_slide(
    polys: List[Polygon],
    other_polys_list: List[List[Polygon]],
    sheet_bounds: Rect,
    sheet_poly: Polygon,
    axis: str,
    spacing: float,
    scale: int,
) -> float:
    """
    Find the maximum distance a part can slide in the negative direction
    of the given axis without overlapping other parts or leaving the sheet.
    Uses binary search with actual polygon intersection checks.
    """
    sheet_min_x, sheet_min_y, sheet_max_x, sheet_max_y = sheet_bounds

    bounds = polygon_group_bounds(polys)
    if axis == "x":
        current_min = bounds[0]
        limit = sheet_min_x + spacing
    else:
        current_min = bounds[1]
        limit = sheet_min_y + spacing

    max_slide = current_min - limit
    if max_slide < 0.01:
        return 0

    best_slide = 0.0
    step = max_slide

    while step > 0.1:
        test_slide = best_slide + step
        if test_slide > max_slide:
            step /= 2
            continue

        if axis == "x":
            test_polys = translate_polygons(polys, -test_slide, 0)
        else:
            test_polys = translate_polygons(polys, 0, -test_slide)

        if _any_overlap(test_polys, other_polys_list):
            step /= 2
        elif not _is_contained(test_polys, sheet_poly, scale):
            step /= 2
        else:
            best_slide = test_slide

    return best_slide


def place_parts(
    parts: List[Dict[str, Any]],
    sheets: List[SheetInfo],
    rotations: List[float],
    config: NestConfig,
    sheet_spacing: float = 0.0,
) -> Optional[NestResult]:
    """
    Place parts using direct intersection checking.
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

    sheet_spatial_grids = {
        sheet.uid: SpatialGrid(cell_size=50.0) for sheet in sheets
    }
    sheet_placed_polys: Dict[str, List[List[Polygon]]] = {
        sheet.uid: [] for sheet in sheets
    }

    part_areas = []
    for i, part in enumerate(parts):
        polygons = part.get("polygons", [])
        area = sum(abs(polygon_area(poly)) for poly in polygons)
        part_areas.append((area, i))

    sorted_indices = [idx for _, idx in sorted(part_areas, reverse=True)]

    placements: List[Placement] = []

    spacing = getattr(config, "spacing", 0.5)

    for sorted_idx in sorted_indices:
        i = sorted_idx
        rotation = rotations[i]
        part = parts[i]
        polygons = part.get("polygons", [])
        uid = part.get("uid", f"part_{i}")

        if not polygons:
            logger.warning("Part '%s' has no polygons", uid)
            continue

        rotated = rotate_polygons(polygons, rotation)
        normalized, orig_min_x, orig_min_y = normalize_polygons(rotated)

        part_bounds = polygon_group_bounds(normalized)
        part_width = part_bounds[2] - part_bounds[0]
        part_height = part_bounds[3] - part_bounds[1]

        best_placement = None
        best_score = float("inf")
        sheets_with_valid_ifp = 0

        for sheet in sheets:
            sheet_bounds = polygon_bounds(sheet.polygon)
            sheet_width = sheet_bounds[2] - sheet_bounds[0]
            sheet_height = sheet_bounds[3] - sheet_bounds[1]

            if part_width > sheet_width or part_height > sheet_height:
                logger.debug(
                    "Part '%s' too large for sheet '%s': "
                    "part(%.2f x %.2f) > sheet(%.2f x %.2f)",
                    uid,
                    sheet.uid,
                    part_width,
                    part_height,
                    sheet_width,
                    sheet_height,
                )
                continue

            ifps = _get_combined_ifp(sheet.polygon, normalized, config)
            if not ifps:
                logger.debug(
                    "Part '%s' has no valid IFP on sheet '%s'",
                    uid,
                    sheet.uid,
                )
                continue

            sheets_with_valid_ifp += 1

            best_pos_for_sheet = None
            best_score_for_sheet = float("inf")

            for ifp in ifps:
                pos = _find_valid_position(
                    ifp,
                    normalized,
                    sheet_placed_polys[sheet.uid],
                    config,
                    spacing,
                    spatial_grid=sheet_spatial_grids[sheet.uid],
                    sheet_world_offset=(
                        sheet.world_offset_x,
                        sheet.world_offset_y,
                    ),
                )

                if pos is not None:
                    rel_x = pos[0] - sheet.world_offset_x
                    rel_y = pos[1] - sheet.world_offset_y

                    if config.placement_type == "gravity":
                        score = rel_y + rel_x * 0.001
                    else:
                        score = rel_x + rel_y * 0.001

                    if score < best_score_for_sheet:
                        best_score_for_sheet = score
                        best_pos_for_sheet = pos

            if best_pos_for_sheet is not None:
                if best_score_for_sheet < best_score:
                    best_score = best_score_for_sheet
                    best_placement = (best_pos_for_sheet, sheet)

        if best_placement is None:
            logger.warning(
                "Part '%s' could not be placed: "
                "%d sheets with valid IFP, but no valid position found",
                uid,
                sheets_with_valid_ifp,
            )
            continue

        assert best_placement is not None
        best_pos, sheet = best_placement
        world_placed_group = translate_polygons(
            normalized, best_pos[0], best_pos[1]
        )

        placement = Placement(
            id=part.get("id", i),
            source=part.get("source", i),
            uid=uid,
            x=best_pos[0],
            y=best_pos[1],
            rotation=rotation,
            polygons=world_placed_group,
            sheet_uid=sheet.uid,
        )

        logger.debug(
            "place_parts: placed '%s' on sheet '%s' at (%.2f, %.2f)",
            uid,
            sheet.uid,
            best_pos[0],
            best_pos[1],
        )

        placements.append(placement)
        sheet_placed_polys[sheet.uid].append(world_placed_group)
        sheet_spatial_grids[sheet.uid].insert(
            len(sheet_placed_polys[sheet.uid]) - 1,
            polygon_group_bounds(world_placed_group),
        )

    if not placements:
        logger.warning("place_parts: no placements made")
        return None

    # Apply gravity per sheet to tighten local packing
    # We do this individually for each sheet to avoid moving parts between
    # sheets
    for sheet in sheets:
        sheet_placements = [p for p in placements if p.sheet_uid == sheet.uid]
        if not sheet_placements:
            continue

        sheet_world_poly = translate_polygons(
            [sheet.polygon], sheet.world_offset_x, sheet.world_offset_y
        )[0]

        # In-place modification of placement objects
        _apply_gravity(
            sheet_placements, sheet_world_poly, spacing, config.clipper_scale
        )

    fitness = _calculate_fitness(placements, num_parts)

    logger.debug(
        "Placement result: %d/%d placements, fitness=%.4f",
        len(placements),
        num_parts,
        fitness,
    )

    total_area = sum(
        abs(polygon_area(poly)) for p in placements for poly in p.polygons
    )

    return NestResult(
        placements=placements,
        fitness=fitness,
        area_used=total_area,
        sheet_index=0,
    )


def _calculate_fitness(
    placements: List[Placement],
    num_parts: int = 0,
) -> float:
    """
    Calculate fitness with multi-sheet awareness and whitespace penalties.
    Lower is better.
    """
    if not placements:
        return float("inf")

    total_part_area = sum(
        abs(polygon_area(poly)) for p in placements for poly in p.polygons
    )

    if total_part_area < 1e-9:
        return float("inf")

    # Group placements by sheet to calculate bounding boxes per sheet
    # This prevents the "gap between sheets" from destroying the fitness score
    sheet_bounds_map = {}

    for p in placements:
        sid = p.sheet_uid or "unknown"
        px, py, pmax_x, pmax_y = polygon_group_bounds(p.polygons)

        if sid not in sheet_bounds_map:
            # min_x, min_y, max_x, max_y, sum_x, sum_y, count
            sheet_bounds_map[sid] = [
                float("inf"),
                float("inf"),
                float("-inf"),
                float("-inf"),
                0.0,
                0.0,
                0,
            ]

        b = sheet_bounds_map[sid]
        b[0] = min(b[0], px)
        b[1] = min(b[1], py)
        b[2] = max(b[2], pmax_x)
        b[3] = max(b[3], pmax_y)

        # Accumulate centroid-like sums for gravity penalty
        b[4] += px
        b[5] += py
        b[6] += 1

    total_bounds_area = 0.0
    gravity_penalty = 0.0
    compaction_penalty = 0.0

    for b in sheet_bounds_map.values():
        width = b[2] - b[0]
        height = b[3] - b[1]

        # Area of the bounding box on this sheet
        total_bounds_area += width * height

        # Penalize width + height (encourages square/compact packing)
        compaction_penalty += width + height

        # Penalize distance from sheet origin (gravity)
        # Sum of (p.x - sheet.min_x) + (p.y - sheet.min_y)
        avg_x_dist = b[4] - (b[0] * b[6])
        avg_y_dist = b[5] - (b[1] * b[6])
        gravity_penalty += avg_x_dist + avg_y_dist

    # Base fitness: ratio of used bounds vs actual part area
    # Ideally 1.0 (perfect fit), practically > 1.0
    fitness = total_bounds_area / total_part_area

    # Add gravity penalty: forces parts to bottom-left to remove whitespace
    # Normalized by sqrt(area) to make it scale-independent relative to bounds
    scale_factor = math.sqrt(total_part_area) if total_part_area > 0 else 1.0
    fitness += (gravity_penalty / scale_factor) * 0.0001

    # Add compaction penalty: minimize perimeter of the pack
    fitness += (compaction_penalty / scale_factor) * 0.001

    # Orientation penalty (prevent unnecessary rotation of rectangles)
    orientation_penalty = 0.0
    for p in placements:
        if len(p.polygons) == 1 and len(p.polygons[0]) == 4:
            rotation_mod = abs(p.rotation) % 90
            if rotation_mod > 45:
                rotation_mod = 90 - rotation_mod
            orientation_penalty += (rotation_mod / 90) * 0.05

    fitness += orientation_penalty

    # Heavy penalty for unplaced parts
    if num_parts > 0 and len(placements) < num_parts:
        missing_penalty = (num_parts - len(placements)) * 1000.0
        fitness += missing_penalty

    return fitness


def validate_placements_no_overlap(
    placements: List[Placement],
    scale: int = 10000000,
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
                            for path in result:
                                area = abs(pyclipper.Area(path))
                                if area > min_overlap_area:
                                    logger.warning(
                                        "Overlap detected: %s vs %s area=%d",
                                        p1.uid[:8],
                                        p2.uid[:8],
                                        area,
                                    )
                                    return False
                    except Exception:
                        pass

    return True
