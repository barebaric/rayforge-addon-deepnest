"""
No-Fit Polygon (NFP) calculation using Minkowski sums.

The NFP is computed as the Minkowski sum of the static polygon A
with the negation of the orbiting polygon B. This allows determining
all valid relative positions where B can be placed without overlapping A.

Reference: minkowski.cc from Deepnest project
"""

from typing import List, Optional
import pyclipper

from rayforge.core.geo.polygon import (
    Polygon,
    Point,
    IntPolygon,
    polygon_bounds,
    int_polygon_bounds,
    almost_equal,
    to_clipper,
    from_clipper,
)
from rayforge.core.geo.minkowski import (
    convolve_point_sequences,
    calculate_input_scale,
)
from .models import NestConfig


def no_fit_polygon(
    static: Polygon, orbiting: Polygon, inside: bool, config: NestConfig
) -> List[Polygon]:
    """
    Calculate the No-Fit Polygon (NFP) for two polygons.

    The NFP represents all valid relative positions where the orbiting
    polygon can be placed without overlapping the static polygon.

    Args:
        static: The fixed polygon (e.g., already placed part or sheet)
        orbiting: The polygon being placed
        inside: If True, calculate inner fit polygon
            (for placement inside container)
        config: Nesting configuration

    Returns:
        List of NFP polygons
    """
    if not static or not orbiting or len(static) < 3 or len(orbiting) < 3:
        return []

    scale = calculate_input_scale([static, orbiting])

    try:
        static_path = to_clipper(static, int(scale))
        orbiting_path = to_clipper(orbiting, int(scale))

        if len(static_path) < 3 or len(orbiting_path) < 3:
            return []

        if inside:
            return _nfp_inside(static_path, orbiting_path, int(scale))
        else:
            return _nfp_outside(static_path, orbiting_path, int(scale))
    except Exception as e:
        logger = __import__("logging").getLogger(__name__)
        logger.debug("NFP calculation failed: %s", e)
        return []


def _nfp_inside(
    static: IntPolygon,
    orbiting: IntPolygon,
    scale: int,
) -> List[Polygon]:
    """
    Calculate Inner Fit Polygon (IFP) for placing orbiting inside static.
    The IFP is the Minkowski difference of the container and the part.

    Matches the logic in minkowski.cc for inside NFP calculation.
    """
    static_min_x, static_min_y, static_max_x, static_max_y = (
        int_polygon_bounds(static)
    )
    orbiting_min_x, orbiting_min_y, orbiting_max_x, orbiting_max_y = (
        int_polygon_bounds(orbiting)
    )

    static_width = static_max_x - static_min_x
    static_height = static_max_y - static_min_y
    orbiting_width = orbiting_max_x - orbiting_min_x
    orbiting_height = orbiting_max_y - orbiting_min_y

    if (
        orbiting_width > static_width + 1
        or orbiting_height > static_height + 1
    ):
        return []

    x_shift = orbiting[0][0]
    y_shift = orbiting[0][1]

    orbiting_negated = [(-p[0], -p[1]) for p in orbiting]

    subjects = []

    parallelograms = convolve_point_sequences(static, orbiting_negated)
    subjects.extend(parallelograms)

    static_shifted = [
        (p[0] + orbiting_negated[0][0], p[1] + orbiting_negated[0][1])
        for p in static
    ]
    subjects.append(static_shifted)

    orbiting_neg_shifted = [
        (p[0] + static[0][0], p[1] + static[0][1]) for p in orbiting_negated
    ]
    subjects.append(orbiting_neg_shifted)

    if not subjects:
        return []

    clipper = pyclipper.Pyclipper()

    for subj in subjects:
        if len(subj) >= 3:
            try:
                clipper.AddPath(subj, pyclipper.PT_SUBJECT, True)
            except Exception:
                continue

    try:
        solution = clipper.Execute(
            pyclipper.CT_UNION,
            pyclipper.PFT_NONZERO,
            pyclipper.PFT_NONZERO,
        )
    except Exception:
        return []

    if not solution:
        return []

    results = []
    for path in solution:
        if len(path) >= 3:
            shifted = [(p[0] + x_shift, p[1] + y_shift) for p in path]
            result_poly = from_clipper(shifted, scale)
            area = pyclipper.Area(path)
            if area > 0:
                results.append(result_poly)

    return results


def _nfp_outside(
    static: IntPolygon, orbiting: IntPolygon, scale: int
) -> List[Polygon]:
    """
    Calculate NFP for placing orbiting outside static (avoiding overlap).
    This is used when placing subsequent parts after the first.

    The NFP represents positions where the reference point of orbiting
    CAN be placed without overlapping static. Positions INSIDE the NFP
    are valid (no overlap), positions OUTSIDE would cause overlap or
    separation.

    This uses the Minkowski difference approach.
    """
    x_shift = orbiting[0][0]
    y_shift = orbiting[0][1]

    orbiting_negated = [(-p[0], -p[1]) for p in orbiting]

    subjects = []

    parallelograms = convolve_point_sequences(static, orbiting_negated)
    subjects.extend(parallelograms)

    static_shifted = [
        (p[0] + orbiting_negated[0][0], p[1] + orbiting_negated[0][1])
        for p in static
    ]
    subjects.append(static_shifted)

    orbiting_neg_shifted = [
        (p[0] + static[0][0], p[1] + static[0][1]) for p in orbiting_negated
    ]
    subjects.append(orbiting_neg_shifted)

    if not subjects:
        return []

    clipper = pyclipper.Pyclipper()

    for subj in subjects:
        if len(subj) >= 3:
            try:
                clipper.AddPath(subj, pyclipper.PT_SUBJECT, True)
            except Exception:
                continue

    try:
        solution = clipper.Execute(
            pyclipper.CT_UNION,
            pyclipper.PFT_NONZERO,
            pyclipper.PFT_NONZERO,
        )
    except Exception:
        return []

    if not solution:
        return []

    results = []
    for path in solution:
        if len(path) >= 3:
            shifted = [(p[0] + x_shift, p[1] + y_shift) for p in path]
            result_poly = from_clipper(shifted, scale)
            area = pyclipper.Area(path)
            if area > 0:
                results.append(result_poly)

    return results


def inner_fit_polygon(
    bin_polygon: Polygon, part_polygon: Polygon, config: NestConfig
) -> Optional[Polygon]:
    """
    Calculate the Inner Fit Polygon (IFP) for placing a part inside a bin.

    The IFP represents all positions where the reference point of the part
    can be placed such that the entire part fits inside the bin.

    For a rectangular case, this is simply the bin shrunk by the part's
    width and height.

    Args:
        bin_polygon: The container (sheet/stock)
        part_polygon: The part to be placed
        config: Nesting configuration

    Returns:
        The IFP polygon, or None if no valid placement area exists.
    """
    if not bin_polygon or len(bin_polygon) < 3:
        return None
    if not part_polygon or len(part_polygon) < 3:
        return None

    bin_bounds = polygon_bounds(bin_polygon)
    part_bounds = polygon_bounds(part_polygon)

    bin_min_x, bin_min_y, bin_max_x, bin_max_y = bin_bounds
    part_min_x, part_min_y, part_max_x, part_max_y = part_bounds

    part_width = part_max_x - part_min_x
    part_height = part_max_y - part_min_y

    bin_width = bin_max_x - bin_min_x
    bin_height = bin_max_y - bin_min_y

    if part_width > bin_width + 1e-6 or part_height > bin_height + 1e-6:
        return None

    ifp_min_x = bin_min_x - part_min_x
    ifp_min_y = bin_min_y - part_min_y
    ifp_max_x = bin_max_x - part_max_x
    ifp_max_y = bin_max_y - part_max_y

    if ifp_min_x > ifp_max_x or ifp_min_y > ifp_max_y:
        return None

    ifp = [
        (ifp_min_x, ifp_min_y),
        (ifp_max_x, ifp_min_y),
        (ifp_max_x, ifp_max_y),
        (ifp_min_x, ifp_max_y),
    ]

    return ifp


def get_placement_position(
    nfp: Polygon,
    part: Polygon,
    position: Point,
    config: NestConfig,
) -> Optional[Point]:
    """
    Find the best placement position within an NFP.

    Args:
        nfp: The No-Fit Polygon defining valid placement area
        part: The part to place (for size reference)
        position: Current position (unused, kept for API compatibility)
        config: Nesting configuration

    Returns:
        Best placement point, or None if no valid position exists
    """
    if not nfp or len(nfp) < 3:
        return None

    min_x, min_y, max_x, max_y = polygon_bounds(part)

    part_width = max_x - min_x
    part_height = max_y - min_y

    if config.placement_type == "gravity":
        return _gravity_placement(nfp, part_width, part_height, config)
    else:
        return _box_placement(nfp, part_width, part_height, config)


def _gravity_placement(
    nfp: Polygon,
    part_width: float,
    part_height: float,
    config: NestConfig,
) -> Optional[Point]:
    """
    Find placement that minimizes Y first, then X (gravity effect).
    This tends to pack parts toward the bottom-left.
    """
    best_point = None
    best_y = float("inf")
    best_x = float("inf")

    for pt in nfp:
        if pt[1] < best_y or (almost_equal(pt[1], best_y) and pt[0] < best_x):
            best_y = pt[1]
            best_x = pt[0]
            best_point = pt

    if best_point is None:
        return None

    return (best_point[0], best_point[1])


def _box_placement(
    nfp: Polygon,
    part_width: float,
    part_height: float,
    config: NestConfig,
) -> Optional[Point]:
    """
    Find placement that minimizes X + Y (bottom-left corner packing).
    """
    best_point = None
    best_score = float("inf")

    for pt in nfp:
        score = pt[0] + pt[1]
        if score < best_score:
            best_score = score
            best_point = pt

    if best_point is None:
        return None

    return (best_point[0], best_point[1])
