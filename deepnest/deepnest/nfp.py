"""
No-Fit Polygon (NFP) calculation using Minkowski sums.

The NFP is computed as the Minkowski sum of the static polygon A
with the negation of the orbiting polygon B. This allows determining
all valid relative positions where B can be placed without overlapping A.

Reference: minkowski.cc from Deepnest project
"""

import logging
import threading
from typing import List, Optional, Tuple, Dict
import pyclipper

from rayforge.core.geo.polygon import (
    Polygon,
    Point,
    IntPolygon,
    polygon_bounds,
    almost_equal,
    to_clipper,
    from_clipper,
)
from rayforge.core.geo.minkowski import (
    convolve_point_sequences,
    calculate_input_scale,
)
from .models import NestConfig

logger = logging.getLogger(__name__)

# --- CACHE SETUP ---
_cache_lock = threading.Lock()
_NFP_CACHE: Dict[
    Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]],
    List[Polygon],
] = {}
_IFP_CACHE: Dict[
    Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]],
    Optional[Polygon],
] = {}


def clear_nfp_cache():
    """Clears the memoization caches for NFP/IFP calculations."""
    with _cache_lock:
        _NFP_CACHE.clear()
        _IFP_CACHE.clear()


def _poly_to_key(poly: Polygon) -> Tuple[Tuple[float, float], ...]:
    """Converts a polygon to a robust hashable key by rounding coordinates."""
    return tuple((round(p[0], 4), round(p[1], 4)) for p in poly)


def _normalize_poly(poly: Polygon) -> Tuple[Polygon, float, float]:
    """Shifts a polygon so its bounding box minimum is at (0, 0)."""
    if not poly:
        return poly, 0.0, 0.0
    min_x = min(p[0] for p in poly)
    min_y = min(p[1] for p in poly)
    norm = [(p[0] - min_x, p[1] - min_y) for p in poly]
    return norm, min_x, min_y


def _nfp_minkowski(
    static: IntPolygon, orbiting: IntPolygon, scale: int
) -> List[Polygon]:
    """
    Calculate NFP using Minkowski sum based on minkowski.cc logic.
    NFP = A + (-B), where A is static and B is orbiting.
    This gives the boundary where B's reference point can be placed for B
    to touch A from the outside.
    """
    x_shift = orbiting[0][0]
    y_shift = orbiting[0][1]

    orbiting_negated = [(-p[0], -p[1]) for p in orbiting]

    subjects = []

    # 1. Edge-edge convolutions (generates parallelograms)
    parallelograms = convolve_point_sequences(static, orbiting_negated)
    subjects.extend(parallelograms)

    # 2. Shifted polygons for reference (handles cases where one polygon
    # is contained entirely within a single feature of the other)
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

    # 3. Perform union of all generated shapes to get the final NFP
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

    # 4. Post-process: shift result to orbiting's reference frame and
    # scale down
    results = []
    for path in solution:
        if len(path) >= 3:
            shifted = [(p[0] + x_shift, p[1] + y_shift) for p in path]
            result_poly = from_clipper(shifted, scale)

            area = pyclipper.Area(path)
            if area > 0:
                results.append(result_poly)

    return results


def no_fit_polygon(
    static: Polygon, orbiting: Polygon, inside: bool, config: NestConfig
) -> List[Polygon]:
    """
    Calculate the No-Fit Polygon (NFP) for two polygons with Caching.
    """
    if not static or not orbiting or len(static) < 3 or len(orbiting) < 3:
        return []

    # 1. Normalize both polygons to origin to maximize cache hits
    norm_static, sx, sy = _normalize_poly(static)
    norm_orbiting, ox, oy = _normalize_poly(orbiting)

    # 2. Build cache key
    s_key = _poly_to_key(norm_static)
    o_key = _poly_to_key(norm_orbiting)
    cache_key = (s_key, o_key)

    # 3. Check Cache
    base_nfps = None
    with _cache_lock:
        if cache_key in _NFP_CACHE:
            base_nfps = _NFP_CACHE[cache_key]

    # 4. Compute if missing
    if base_nfps is None:
        scale = calculate_input_scale([norm_static, norm_orbiting])
        try:
            static_path = to_clipper(norm_static, int(scale))
            orbiting_path = to_clipper(norm_orbiting, int(scale))

            if len(static_path) < 3 or len(orbiting_path) < 3:
                base_nfps = []
            else:
                base_nfps = _nfp_minkowski(
                    static_path, orbiting_path, int(scale)
                )
        except Exception as e:
            logger.debug("NFP calculation failed: %s", e)
            base_nfps = []

        with _cache_lock:
            _NFP_CACHE[cache_key] = base_nfps

    if not base_nfps:
        return []

    # 5. Translate cached result to world coordinates
    # Because Minkowski sum is translation-invariant:
    #   NFP(A_world, B) = NFP(A_norm, B) + Translation_A
    translated_nfps = []
    for nfp in base_nfps:
        translated_nfps.append([(p[0] + sx, p[1] + sy) for p in nfp])

    return translated_nfps


def inner_fit_polygon(
    bin_polygon: Polygon, part_polygon: Polygon, config: NestConfig
) -> Optional[Polygon]:
    """
    Calculate the Inner Fit Polygon (IFP) for placing a part inside a bin.
    """
    if not bin_polygon or len(bin_polygon) < 3:
        return None
    if not part_polygon or len(part_polygon) < 3:
        return None

    # 1. Normalize
    norm_bin, bx, by = _normalize_poly(bin_polygon)
    norm_part, px, py = _normalize_poly(part_polygon)

    # 2. Cache Key
    b_key = _poly_to_key(norm_bin)
    p_key = _poly_to_key(norm_part)
    cache_key = (b_key, p_key)

    base_ifp = False  # Use False to distinguish from None (which means no fit)
    with _cache_lock:
        if cache_key in _IFP_CACHE:
            base_ifp = _IFP_CACHE[cache_key]

    if base_ifp is False:
        bin_bounds = polygon_bounds(norm_bin)
        part_bounds = polygon_bounds(norm_part)

        bin_min_x, bin_min_y, bin_max_x, bin_max_y = bin_bounds
        part_min_x, part_min_y, part_max_x, part_max_y = part_bounds

        part_width = part_max_x - part_min_x
        part_height = part_max_y - part_min_y

        bin_width = bin_max_x - bin_min_x
        bin_height = bin_max_y - bin_min_y

        if part_width > bin_width + 1e-6 or part_height > bin_height + 1e-6:
            base_ifp = None
        else:
            ifp_min_x = bin_min_x - part_min_x
            ifp_min_y = bin_min_y - part_min_y
            ifp_max_x = bin_max_x - part_max_x
            ifp_max_y = bin_max_y - part_max_y

            if ifp_min_x > ifp_max_x or ifp_min_y > ifp_max_y:
                base_ifp = None
            else:
                base_ifp = [
                    (ifp_min_x, ifp_min_y),
                    (ifp_max_x, ifp_min_y),
                    (ifp_max_x, ifp_max_y),
                    (ifp_min_x, ifp_max_y),
                ]

        with _cache_lock:
            _IFP_CACHE[cache_key] = base_ifp

    if base_ifp is None:
        return None

    # 3. Translate back to world coordinates
    # For IFP, the locus shift is exactly (T_bin - T_part)
    tx = bx - px
    ty = by - py
    return [(pt[0] + tx, pt[1] + ty) for pt in base_ifp]


def get_placement_position(
    nfp: Polygon,
    part: Polygon,
    position: Point,
    config: NestConfig,
) -> Optional[Point]:
    """
    Find the best placement position within an NFP.
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
