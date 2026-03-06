from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from rayforge.core.geo.polygon import Polygon


@dataclass
class NestConfig:
    curve_tolerance: float = 0.05
    simplify_tolerance: float = 0.1
    spacing: float = 0.0
    rotations: int = 36
    population_size: int = 10
    mutation_rate: int = 30
    placement_type: Literal["gravity", "box"] = "gravity"
    merge_lines: bool = True
    simplify: bool = False
    scale: float = 1.0
    clipper_scale: int = 10000000
    time_ratio: float = 0.5
    overlap_tolerance: float = 0.0001


@dataclass
class WorkpieceInfo:
    uid: str
    polygons: List[Polygon]
    source: int
    quantity: int = 1
    is_sheet: bool = False
    offset_x: float = 0.0
    offset_y: float = 0.0
    # Pre-computed convex hulls for hierarchical collision detection
    hulls: List[Polygon] = field(default_factory=list)


@dataclass
class SheetInfo:
    """Represents a single sheet in unified world space."""

    uid: str
    polygon: Polygon
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
    polygons: List[Polygon]
    sheet_uid: Optional[str] = None
    # Rotated convex hulls corresponding to the polygons
    hulls: List[Polygon] = field(default_factory=list)


@dataclass
class NestSolution:
    placements: List[Dict[str, Any]]
    fitness: float
    area_used: float
