import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pyclipper

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
)
from rayforge.core.geo.query import bboxes_intersect
from .models import NestConfig, Placement, SheetInfo
from .nfp import inner_fit_polygon
from .spatial_grid import SpatialGrid

logger = logging.getLogger(__name__)


@dataclass
class NestResult:
    placements: List[Placement]
    fitness: float
    area_used: float
    sheet_index: int = 0


@dataclass
class MultiSheetResult:
    """Result of multi-sheet nesting."""

    placements: List[Placement]
    fitness: float
    sheets_used: List[str]
    unplaced_count: int


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


def _get_combined_ifp(
    sheet: Polygon,
    part_polygons: List[Polygon],
    config: NestConfig,
) -> Optional[Polygon]:
    """Get the combined IFP for all polygons in a part using intersection."""
    if not part_polygons:
        return None

    combined_ifp = None
    scale = config.clipper_scale

    for poly in part_polygons:
        ifp = inner_fit_polygon(sheet, poly, config)
        if ifp is None:
            return None

        if combined_ifp is None:
            combined_ifp = ifp
        else:
            try:
                clipper = pyclipper.Pyclipper()
                clipper.AddPath(
                    to_clipper(combined_ifp, scale),
                    pyclipper.PT_CLIP,
                    True,
                )
                clipper.AddPath(
                    to_clipper(ifp, scale),
                    pyclipper.PT_SUBJECT,
                    True,
                )
                result = clipper.Execute(
                    pyclipper.CT_INTERSECTION,
                    pyclipper.PFT_NONZERO,
                    pyclipper.PFT_NONZERO,
                )
                if result:
                    combined_ifp = from_clipper(result[0], scale)
                else:
                    return None
            except Exception:
                return None

    return combined_ifp


def _find_valid_position(
    ifp: Polygon,
    part_polygons: List[Polygon],
    placed_polys_list: List[List[Polygon]],
    config: NestConfig,
    spacing: float = 0.1,
    edge_sample_step: float = 5.0,
    spatial_grid: Optional[SpatialGrid] = None,
    sheet_world_offset: Tuple[float, float] = (0.0, 0.0),
) -> Optional[Tuple[float, float]]:
    """
    Find a valid position in the IFP where part doesn't overlap placed parts.
    Uses bottom-left fill strategy with edge sampling for candidates.
    Implements early termination and exposed-edge-only sampling for speed.

    All placed_polys_list entries are in world coordinates.
    sheet_world_offset is used to convert IFP to world coordinates
    for overlap checking.
    """
    if not ifp or len(ifp) < 3:
        return None

    offset_x, offset_y = sheet_world_offset
    ifp_world = translate_polygons([ifp], offset_x, offset_y)[0]

    ifp_bounds = polygon_bounds(ifp_world)
    ifp_min_x, ifp_min_y, ifp_max_x, ifp_max_y = ifp_bounds

    part_bounds = polygon_group_bounds(part_polygons)
    part_min_x, part_min_y, part_max_x, part_max_y = part_bounds

    candidates = []

    for pt in ifp:
        candidates.append((pt[0] + offset_x, pt[1] + offset_y))

    if spatial_grid is not None and placed_polys_list:
        expanded_bbox = (
            ifp_min_x - edge_sample_step,
            ifp_min_y - edge_sample_step,
            ifp_max_x + edge_sample_step,
            ifp_max_y + edge_sample_step,
        )
        nearby_indices = spatial_grid.query(expanded_bbox)
        parts_to_sample = [
            placed_polys_list[i]
            for i in nearby_indices
            if i < len(placed_polys_list)
        ]
    else:
        parts_to_sample = placed_polys_list

    for placed_polys in parts_to_sample:
        bbox = polygon_group_bounds(placed_polys)
        min_x, min_y, max_x, max_y = bbox

        right_x = max_x + spacing
        top_y = max_y + spacing

        num_x_samples = max(2, int((max_x - min_x) / edge_sample_step))
        num_y_samples = max(2, int((max_y - min_y) / edge_sample_step))

        for i in range(num_x_samples + 1):
            t = i / num_x_samples
            x = min_x + t * (max_x - min_x)
            candidates.append((x, top_y))

        for i in range(num_y_samples + 1):
            t = i / num_y_samples
            y = min_y + t * (max_y - min_y)
            candidates.append((right_x, y))

        for poly in placed_polys:
            for pt in poly:
                candidates.append((pt[0] + spacing, pt[1]))
                candidates.append((pt[0], pt[1] + spacing))

    best_score = float("inf")
    best_pos = None

    for x, y in candidates:
        if x < ifp_min_x - 0.01 or x > ifp_max_x + 0.01:
            continue
        if y < ifp_min_y - 0.01 or y > ifp_max_y + 0.01:
            continue

        if not point_in_polygon((x, y), ifp_world):
            continue

        if config.placement_type == "gravity":
            score = y + x * 0.001
        else:
            score = x + y * 0.001

        if score >= best_score:
            continue

        test_polys = translate_polygons(part_polygons, x, y)
        cand_bbox = (
            x + part_min_x,
            y + part_min_y,
            x + part_max_x,
            y + part_max_y,
        )
        if _any_overlap(
            test_polys, placed_polys_list, spatial_grid, cand_bbox
        ):
            continue

        best_score = score
        best_pos = (x, y)

    return best_pos


def _apply_gravity(
    placements: List[Placement],
    sheet_bounds: Rect,
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

    if not other_polys_list:
        return max_slide

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

            ifp = _get_combined_ifp(sheet.polygon, normalized, config)
            if ifp is None:
                logger.debug(
                    "Part '%s' has no valid IFP on sheet '%s'",
                    uid,
                    sheet.uid,
                )
                continue

            sheets_with_valid_ifp += 1

            best_pos = _find_valid_position(
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

            if best_pos is None:
                continue

            rel_x = best_pos[0] - sheet.world_offset_x
            rel_y = best_pos[1] - sheet.world_offset_y

            if config.placement_type == "gravity":
                score = rel_y + rel_x * 0.001
            else:
                score = rel_x + rel_y * 0.001

            if score < best_score:
                best_score = score
                best_placement = (best_pos, sheet)

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
        placements.append(placement)
        sheet_placed_polys[sheet.uid].append(world_placed_group)
        sheet_spatial_grids[sheet.uid].insert(
            len(sheet_placed_polys[sheet.uid]) - 1,
            polygon_group_bounds(world_placed_group),
        )

    if not placements:
        logger.warning("place_parts: no placements made")
        return None

    combined_bounds = None
    for sheet in sheets:
        bounds = polygon_bounds(sheet.polygon)
        offset_bounds = (
            bounds[0] + sheet.world_offset_x,
            bounds[1] + sheet.world_offset_y,
            bounds[2] + sheet.world_offset_x,
            bounds[3] + sheet.world_offset_y,
        )
        if combined_bounds is None:
            combined_bounds = offset_bounds
        else:
            combined_bounds = (
                min(combined_bounds[0], offset_bounds[0]),
                min(combined_bounds[1], offset_bounds[1]),
                max(combined_bounds[2], offset_bounds[2]),
                max(combined_bounds[3], offset_bounds[3]),
            )

    if combined_bounds is not None and len(sheets) == 1:
        placements = _apply_gravity(
            placements, combined_bounds, spacing, config.clipper_scale
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
    if not placements:
        return float("inf")

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for p in placements:
        px, py, pmax_x, pmax_y = polygon_group_bounds(p.polygons)
        min_x = min(min_x, px)
        min_y = min(min_y, py)
        max_x = max(max_x, pmax_x)
        max_y = max(max_y, pmax_y)

    bounding_area = (max_x - min_x) * (max_y - min_y)

    total_part_area = sum(
        abs(polygon_area(poly)) for p in placements for poly in p.polygons
    )

    if total_part_area < 1e-9:
        return float("inf")

    utilization = total_part_area / bounding_area if bounding_area > 0 else 0

    fitness = 1.0 / utilization if utilization > 0 else float("inf")

    # Shape orientation penalty - penalize rotated rectangles to keep
    # them in natural orientation. This helps prevent unnecessary
    # 90-degree rotations of simple rectangular shapes.
    orientation_penalty = 0.0
    for p in placements:
        # Check if the shape is approximately rectangular
        if len(p.polygons) == 1:
            poly = p.polygons[0]
            if len(poly) == 4:
                # Calculate width and height of the polygon
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)

                # If it's close to a rectangle, penalize rotation from
                # natural orientation
                aspect_ratio = width / height if height > 0 else 0
                if 0.5 <= aspect_ratio <= 2.0:  # Roughly square or rectangular
                    # Calculate how far the rotation is from 0 or 90 degrees
                    rotation_mod = abs(p.rotation) % 90
                    if rotation_mod > 45:
                        rotation_mod = 90 - rotation_mod

                    # Apply penalty for being away from natural orientation
                    # Stronger penalty for larger deviations
                    orientation_penalty += (rotation_mod / 90) * 0.1

    fitness += orientation_penalty

    # Remove height penalty - it can cause counterintuitive results
    # height_penalty = max_y * 0.001
    # fitness += height_penalty

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
