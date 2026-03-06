from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class NestConfig:
    """
    Configuration parameters for the nesting algorithm.

    Attributes:
        curve_tolerance: Tolerance for converting geometry curves to polygon
            approximations (used in Geometry.to_polygons()). Lower values
            produce more accurate curves but increase computation time.
        simplify_tolerance: Tolerance for Ramer-Douglas-Peucker polygon
            simplification. Higher values reduce vertex count more aggressively
            but may lose detail.
        spacing: Distance to offset each part outward before nesting, creating
            gaps between placed parts. Set to 0 for tight packing.
        rotations: Number of discrete rotation angles to evaluate. The angular
            step is 360/rotations degrees (e.g., 36 = 10° steps).
        population_size: Number of candidate solutions (individuals) in the
            genetic algorithm population. Larger populations explore more
            possibilities but take longer per generation.
        mutation_rate: Probability of mutation in the genetic algorithm,
            expressed as a percentage (value/100). E.g., 30 = 30% chance.
        merge_lines: If True, merges collinear line segments in polygons,
            reducing vertex count for simpler geometry.
        simplify: If True, replaces each polygon with its convex hull for
            faster but less accurate nesting.
        clipper_scale: Integer scaling factor for Clipper library operations.
            Higher values increase precision for small features but may impact
            performance. Default 10000000 provides good precision for most
            cases.
        target_utilization: Target material utilization (0.0-1.0). Nesting
            stops early if efficiency exceeds this value. Fitness is
            1/utilization, so 0.99 means stop when fitness < ~1.01.
    """

    curve_tolerance: float = 0.05
    simplify_tolerance: float = 0.1
    spacing: float = 0.0
    rotations: int = 36
    population_size: int = 10
    mutation_rate: int = 30
    merge_lines: bool = True
    simplify: bool = False
    clipper_scale: int = 10000000
    target_utilization: float = 0.99


@dataclass
class WorkpieceInfo:
    uid: str
    polygons: List[np.ndarray]
    source: int
    quantity: int = 1
    is_sheet: bool = False
    offset_x: float = 0.0
    offset_y: float = 0.0
    # Pre-computed convex hulls for hierarchical collision detection
    hulls: List[np.ndarray] = field(default_factory=list)


@dataclass
class SheetInfo:
    """Represents a single sheet in unified world space."""

    uid: str
    polygon: np.ndarray
    world_offset_x: float = 0.0
    world_offset_y: float = 0.0


@dataclass
class Placement:
    id: int
    source: int
    uid: str
    x: float
    y: float
    rotation: float
    polygons: List[np.ndarray]
    sheet_uid: Optional[str] = None
    # Rotated convex hulls corresponding to the polygons
    hulls: List[np.ndarray] = field(default_factory=list)


@dataclass
class NestSolution:
    placements: List[Dict[str, Any]]
    fitness: float
    area_used: float
