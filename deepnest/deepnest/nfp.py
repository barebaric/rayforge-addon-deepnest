"""
No-Fit Polygon (NFP) calculation using Minkowski sums.

The NFP is computed as the Minkowski sum of the static polygon A
with the negation of the orbiting polygon B. This allows determining
all valid relative positions where B can be placed without overlapping A.

Reference: minkowski.cc from Deepnest project
"""

import logging
import threading
from typing import List, Tuple, Dict
import pyclipper

from rayforge.core.geo.minkowski import (
    convolve_point_sequences,
    minkowski_sum_convex,
)
from rayforge.core.geo.polygon import (
    Polygon,
    IntPolygon,
    to_clipper,
    from_clipper,
    is_convex,
    clean_polygon,
    convex_hull,
    polygon_bounds,
)
from .models import NestConfig

logger = logging.getLogger(__name__)

# --- CACHE SETUP ---
# Limit cache size to prevent memory leaks during long GA runs with many
# rotations
MAX_CACHE_SIZE = 2000

_cache_lock = threading.Lock()
_NFP_CACHE: Dict[
    Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]],
    List[Polygon],
] = {}
_IFP_CACHE: Dict[
    Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]],
    List[Polygon],
] = {}


def clear_nfp_cache():
    """Clears the memoization caches for NFP/IFP calculations."""
    with _cache_lock:
        _NFP_CACHE.clear()
        _IFP_CACHE.clear()


def _manage_cache_size(cache: Dict):
    """Enforce hard limit on cache size to prevent infinite growth."""
    if len(cache) > MAX_CACHE_SIZE:
        cache.clear()


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


def _nfp_convex_fast(
    static: IntPolygon, orbiting: IntPolygon, scale: int
) -> List[Polygon]:
    """
    Fast NFP calculation for convex-convex polygon pairs.

    Uses the simplified Minkowski sum formula for convex polygons:
    NFP = ConvexHull(A + (-B)), which avoids the expensive union operation.
    """
    x_shift = orbiting[0][0]
    y_shift = orbiting[0][1]

    orbiting_negated = [(-p[0], -p[1]) for p in orbiting]

    nfp_paths = minkowski_sum_convex(static, orbiting_negated)

    results = []
    for path in nfp_paths:
        if len(path) >= 3:
            shifted = [(p[0] + x_shift, p[1] + y_shift) for p in path]
            result_poly = from_clipper(shifted, scale)
            results.append(result_poly)

    return results


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
        else:
            _manage_cache_size(_NFP_CACHE)

    # 4. Compute if missing
    if base_nfps is None:
        # Use consistent scaling from config to avoid precision mismatches
        scale = config.clipper_scale
        try:
            static_path = to_clipper(norm_static, scale)
            orbiting_path = to_clipper(norm_orbiting, scale)

            if len(static_path) < 3 or len(orbiting_path) < 3:
                base_nfps = []
            elif is_convex(norm_static) and is_convex(norm_orbiting):
                base_nfps = _nfp_convex_fast(static_path, orbiting_path, scale)
            else:
                base_nfps = _nfp_minkowski(static_path, orbiting_path, scale)
        except Exception as e:
            logger.debug("NFP calculation failed: %s", e)
            base_nfps = []

        with _cache_lock:
            _NFP_CACHE[cache_key] = base_nfps

    if not base_nfps:
        return []

    # 5. Translate cached result to world coordinates
    translated_nfps = []
    for nfp in base_nfps:
        translated_nfps.append([(p[0] + sx, p[1] + sy) for p in nfp])

    return translated_nfps


def inner_fit_polygon(
    bin_polygon: Polygon, part_polygon: Polygon, config: NestConfig
) -> List[Polygon]:
    """
    Calculate the Inner Fit Polygon (IFP) for placing a part inside a bin.
    """
    if not bin_polygon or len(bin_polygon) < 3:
        return []
    if not part_polygon or len(part_polygon) < 3:
        return []

    # Ensure inputs are clean
    clean_bin = clean_polygon(bin_polygon, 0.001)
    if not clean_bin:
        clean_bin = bin_polygon
    clean_part = clean_polygon(part_polygon, 0.001)
    if not clean_part:
        clean_part = part_polygon

    # 1. Normalize
    norm_bin, bx, by = _normalize_poly(clean_bin)
    norm_part, px, py = _normalize_poly(clean_part)

    # 2. Cache Key
    b_key = _poly_to_key(norm_bin)
    p_key = _poly_to_key(norm_part)
    cache_key = (b_key, p_key)

    base_ifps = None
    with _cache_lock:
        if cache_key in _IFP_CACHE:
            base_ifps = _IFP_CACHE[cache_key]
        else:
            _manage_cache_size(_IFP_CACHE)

    if base_ifps is None:
        bin_bounds = polygon_bounds(norm_bin)
        part_bounds = polygon_bounds(norm_part)

        bin_width = bin_bounds[2] - bin_bounds[0]
        bin_height = bin_bounds[3] - bin_bounds[1]
        part_width = part_bounds[2] - part_bounds[0]
        part_height = part_bounds[3] - part_bounds[1]

        # Quick reject if bounding box doesn't fit
        if part_width > bin_width + 1e-4 or part_height > bin_height + 1e-4:
            base_ifps = []
        else:
            scale = config.clipper_scale

            bin_clip = to_clipper(norm_bin, scale)
            part_clip = to_clipper(norm_part, scale)

            # Negate part for Minkowski Sum
            part_neg = [(-p[0], -p[1]) for p in part_clip]

            # Generate Solid No-Go Zones via Convex Hull Sweeps
            # Ensures a solid band around the edge, solving
            # "hollow trace" issues

            clipper_union = pyclipper.Pyclipper()

            # 1. Add Part at every vertex (Corner caps)
            for v in bin_clip:
                translated_part = [
                    (p[0] + v[0], p[1] + v[1]) for p in part_neg
                ]
                clipper_union.AddPath(
                    translated_part, pyclipper.PT_SUBJECT, True
                )

            # 2. Add Convex Hull Sweep for every edge
            # Convert part_neg to float for convex_hull calculation
            part_neg_float = [(float(p[0]), float(p[1])) for p in part_neg]

            for i in range(len(bin_clip)):
                p1 = bin_clip[i - 1]
                p2 = bin_clip[i]

                # Create cloud of points: Part at P1 + Part at P2
                points = []
                for p in part_neg_float:
                    points.append((p[0] + p1[0], p[1] + p1[1]))
                    points.append((p[0] + p2[0], p[1] + p2[1]))

                # The Convex Hull of these points is the solid sweep
                # of the part along the segment
                hull = convex_hull(points)
                hull_int = [(int(p[0]), int(p[1])) for p in hull]

                clipper_union.AddPath(hull_int, pyclipper.PT_SUBJECT, True)

            try:
                # Union of all solid sweeps
                no_go_zones = clipper_union.Execute(
                    pyclipper.CT_UNION,
                    pyclipper.PFT_NONZERO,
                    pyclipper.PFT_NONZERO,
                )
            except Exception as e:
                logger.debug("IFP Union failed: %s", e)
                no_go_zones = None

            if no_go_zones is None:
                base_ifps = []
            else:
                # IFP = Bin - NoGoZones
                clipper_diff = pyclipper.Pyclipper()
                clipper_diff.AddPath(bin_clip, pyclipper.PT_SUBJECT, True)
                if no_go_zones:
                    for nogo in no_go_zones:
                        clipper_diff.AddPath(nogo, pyclipper.PT_CLIP, True)

                try:
                    ifp_solution = clipper_diff.Execute(
                        pyclipper.CT_DIFFERENCE,
                        pyclipper.PFT_NONZERO,
                        pyclipper.PFT_NONZERO,
                    )
                except Exception as e:
                    logger.debug("IFP Difference failed: %s", e)
                    ifp_solution = []

                if ifp_solution:
                    base_ifps = []
                    for path in ifp_solution:
                        if len(path) >= 3:
                            base_ifps.append(from_clipper(path, scale))
                else:
                    base_ifps = []

        with _cache_lock:
            _IFP_CACHE[cache_key] = base_ifps

    if not base_ifps:
        return []

    # 3. Translate back to world coordinates
    tx = bx - px
    ty = by - py
    translated_ifps = []
    for ifp in base_ifps:
        translated_ifps.append([(pt[0] + tx, pt[1] + ty) for pt in ifp])

    return translated_ifps
