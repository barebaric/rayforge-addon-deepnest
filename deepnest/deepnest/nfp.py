"""
No-Fit Polygon (NFP) calculation using Minkowski sums.

The NFP is computed as the Minkowski sum of the static polygon A
with the negation of the orbiting polygon B. This allows determining
all valid relative positions where B can be placed without overlapping A.

Reference: minkowski.cc from Deepnest project
"""

import logging
import threading
from typing import Dict, List, Tuple

from raygeo.geo.shape.polygon import clean_polygon
from raygeo.geo.types import Point, Polygon
from raygeo.nest import ifp, nfp
from .models import NestConfig

logger = logging.getLogger(__name__)

# --- CACHE SETUP ---
# Limit cache size to prevent memory leaks during long GA runs with many
# rotations
MAX_CACHE_SIZE = 2000

_cache_lock = threading.Lock()
_NFP_CACHE: Dict[
    Tuple[Tuple[Point, ...], Tuple[Point, ...]],
    List[Polygon],
] = {}
_IFP_CACHE: Dict[
    Tuple[Tuple[Point, ...], Tuple[Point, ...]],
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


def no_fit_polygon(
    static: Polygon, orbiting: Polygon, inside: bool, config: NestConfig
) -> List[Polygon]:
    """
    Calculate the No-Fit Polygon (NFP) for two polygons with Caching.
    """
    if not static or not orbiting or len(static) < 3 or len(orbiting) < 3:
        return []

    # 1. Normalize both polygons to origin to maximize cache hits
    norm_static, sx, sy = nfp.normalize_polygon(static)
    norm_orbiting, ox, oy = nfp.normalize_polygon(orbiting)

    # 2. Build cache key
    s_key = tuple(nfp.polygon_to_key(norm_static))
    o_key = tuple(nfp.polygon_to_key(norm_orbiting))
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
        try:
            base_nfps = nfp.no_fit_polygon(norm_static, norm_orbiting)
        except Exception as e:
            logger.debug("NFP calculation failed: %s", e)
            base_nfps = []

        with _cache_lock:
            _NFP_CACHE[cache_key] = base_nfps

    if not base_nfps:
        return []

    # 5. Translate cached result to world coordinates
    translated_nfps = []
    for poly in base_nfps:
        translated_nfps.append([(p[0] + sx, p[1] + sy) for p in poly])

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
    norm_bin, bx, by = nfp.normalize_polygon(clean_bin)
    norm_part, px, py = nfp.normalize_polygon(clean_part)

    # 2. Cache Key
    b_key = tuple(nfp.polygon_to_key(norm_bin))
    p_key = tuple(nfp.polygon_to_key(norm_part))
    cache_key = (b_key, p_key)

    base_ifps = None
    with _cache_lock:
        if cache_key in _IFP_CACHE:
            base_ifps = _IFP_CACHE[cache_key]
        else:
            _manage_cache_size(_IFP_CACHE)

    if base_ifps is None:
        try:
            base_ifps = ifp.inner_fit_polygon(norm_bin, norm_part)
        except Exception as e:
            logger.debug("IFP calculation failed: %s", e)
            base_ifps = []

        with _cache_lock:
            _IFP_CACHE[cache_key] = base_ifps

    if not base_ifps:
        return []

    # 3. Translate back to world coordinates
    tx = bx - px
    ty = by - py
    translated_ifps = []
    for poly in base_ifps:
        translated_ifps.append([(pt[0] + tx, pt[1] + ty) for pt in poly])

    return translated_ifps
