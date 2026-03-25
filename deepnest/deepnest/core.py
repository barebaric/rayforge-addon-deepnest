import asyncio
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

from rayforge.core.geo import Geometry
from rayforge.core.geo.polygon import (
    polygon_area_numpy,
    polygon_bounds_numpy,
    polygon_offset,
    clean_polygon,
    convex_hull,
    normalize_polygons,
)
from rayforge.core.geo.simplify import simplify_points
from rayforge.core.geo.types import Polygon
from .genetic import GeneticAlgorithm
from .models import (
    NestConfig,
    NestSolution,
    Placement,
    SheetInfo,
    WorkpieceInfo,
)
from .placement import (
    NestResult,
    place_parts,
)
from .nfp import clear_nfp_cache

if TYPE_CHECKING:
    from rayforge.shared.tasker.manager import TaskManager
    from rayforge.shared.tasker.context import ExecutionContext

logger = logging.getLogger(__name__)


@dataclass
class _AsyncNestingState:
    best_placements: List[Placement] = field(default_factory=list)
    best_fitness: float = float("inf")
    generations_without_improvement: int = 0
    pending_tasks: Dict[str, int] = field(default_factory=dict)
    generation: int = 0
    done: bool = False
    results: List[tuple[str, int, Optional[NestResult]]] = field(
        default_factory=list
    )
    lock: threading.Lock = field(default_factory=threading.Lock)
    improved_this_generation: bool = False


def _place_parts_worker(
    context,
    parts: List[Dict[str, Any]],
    sheets: List[SheetInfo],
    rotations: List[float],
    config: NestConfig,
    flips_h: Optional[List[bool]] = None,
    flips_v: Optional[List[bool]] = None,
) -> Optional[NestResult]:
    return place_parts(
        parts, sheets, rotations, config, flips_h=flips_h, flips_v=flips_v
    )


def _simplify_polygon(polygon: Polygon, config: NestConfig) -> Polygon:
    """Simplify a polygon using RDP or convex hull if configured."""

    # 1. Adaptive Simplification (Ramer-Douglas-Peucker)
    # Reduces vertex count on complex curves before they enter the nesting
    # logic
    if config.simplify_tolerance > 0:
        polygon = simplify_points(polygon, config.simplify_tolerance)

    # 2. Convex Hull Simplification (Optional / Aggressive)
    if config.simplify:
        return convex_hull(polygon)

    # 3. Merge Collinear Segments (Optional)
    if config.merge_lines:
        cleaned = clean_polygon(polygon, 0.1)
        if cleaned:
            polygon = cleaned

    # 4. Topology Cleaning
    # Remove self-intersections and negligible artifacts
    tolerance = 0.01 * config.curve_tolerance
    cleaned = clean_polygon(polygon, tolerance)
    if cleaned:
        return cleaned

    return polygon


class DeepNest:
    def __init__(self, config: Optional[NestConfig] = None):
        self.config = config or NestConfig()
        self._workpieces: List[WorkpieceInfo] = []
        self._sheets: List[WorkpieceInfo] = []
        self._ga: Optional[GeneticAlgorithm] = None
        self._nests: List[NestSolution] = []
        self._working = False
        self._cancelled = False

    def add_geometry(
        self,
        geometry: Geometry,
        uid: str = "",
        quantity: int = 1,
        is_sheet: bool = False,
    ) -> bool:
        if geometry.is_empty():
            logger.warning("add_geometry: geometry is empty for '%s'", uid)
            return False

        polygons = geometry.to_polygons(self.config.curve_tolerance)
        if not polygons:
            logger.warning(
                "add_geometry: to_polygons returned empty for '%s' "
                "(curve_tolerance=%.2f)",
                uid,
                self.config.curve_tolerance,
            )
            return False

        processed = []
        for poly in polygons:
            if self.config.spacing > 0:
                offset_polys = polygon_offset(poly, 0.5 * self.config.spacing)
                if offset_polys:
                    poly = offset_polys[0]
            processed.append(_simplify_polygon(poly, self.config))

        if not processed:
            logger.debug(
                "add_geometry: no polygons after processing for '%s'", uid
            )
            return False

        processed, offset_x, offset_y = normalize_polygons(processed)

        # Convert to numpy arrays for the models
        processed_np = [np.array(p) for p in processed]

        # Pre-calculate convex hulls for hierarchical collision detection
        hulls = [np.array(convex_hull(p)) for p in processed]

        info = WorkpieceInfo(
            uid=uid or f"geo_{len(self._workpieces) + len(self._sheets)}",
            polygons=processed_np,
            source=len(self._workpieces) + len(self._sheets),
            quantity=quantity,
            is_sheet=is_sheet,
            offset_x=offset_x,
            offset_y=offset_y,
            hulls=hulls,
        )

        if is_sheet:
            self._sheets.append(info)
        else:
            self._workpieces.append(info)

        logger.debug(
            "add_geometry: added '%s' with %d polygon(s), is_sheet=%s",
            uid,
            len(processed),
            is_sheet,
        )
        return True

    def add_sheet(self, polygon: np.ndarray, uid: str = "") -> bool:
        """Add a sheet polygon directly.

        Args:
            polygon: The sheet boundary polygon as numpy array
            uid: Optional identifier for the sheet

        Returns:
            True if the sheet was added successfully
        """
        geometry = Geometry()

        # Handle numpy array input
        if isinstance(polygon, np.ndarray):
            polygon = polygon.tolist()

        if not polygon or len(polygon) < 3:
            logger.warning("add_sheet: invalid polygon")
            return False

        geometry.move_to(polygon[0][0], polygon[0][1])
        for pt in polygon[1:]:
            geometry.line_to(pt[0], pt[1])
        geometry.close_path()

        return self.add_geometry(
            geometry,
            uid=uid or f"sheet_{len(self._sheets)}",
            is_sheet=True,
        )

    def clear(self) -> None:
        self._workpieces.clear()
        self._sheets.clear()
        self._ga = None
        self._nests.clear()
        self._cancelled = False

    def nest(self) -> Optional[NestSolution]:
        clear_nfp_cache()

        if not self._workpieces:
            logger.warning("nest: no workpieces to nest")
            return None

        if not self._sheets:
            default_sheet = self._create_default_sheet()
            if default_sheet is None:
                logger.warning("nest: failed to create default sheet")
                return None
            self._sheets.append(default_sheet)
            logger.debug("nest: created default sheet")

        parts = self._prepare_parts()
        if not parts:
            logger.warning("nest: no parts after preparation")
            return None

        num_parts = len(parts)
        if num_parts > 50:
            num_generations = 3
        elif num_parts > 20:
            num_generations = 5
        elif num_parts > 10:
            num_generations = 15
        else:
            # For simple cases, use more generations for better exploration
            num_generations = min(30, max(10, num_parts * 2))

        logger.info(
            "Starting nest: %d part(s), %d generation(s)",
            num_parts,
            num_generations,
        )

        sheets = [
            SheetInfo(
                uid=s.uid,
                polygon=s.polygons[0] if len(s.polygons) > 0 else np.array([]),
                world_offset_x=s.offset_x,
                world_offset_y=s.offset_y,
            )
            for s in self._sheets
        ]

        logger.debug("Using %d sheet(s)", len(sheets))

        self._ga = GeneticAlgorithm(parts, self.config)

        # Calculate target fitness for early termination
        # Fitness is ~ 1/Utilization. Lower fitness is better.
        target_fitness = 0.0
        if self.config.target_utilization > 0:
            target_fitness = 1.0 / self.config.target_utilization

        # First, evaluate the identity placement (original positions/rotations)
        if len(parts) < 50:
            identity_rotations = [0.0] * len(parts)
            identity_result = place_parts(
                parts, sheets, identity_rotations, self.config
            )

            best_solution: Optional[NestSolution] = None
            if identity_result and identity_result.fitness < float("inf"):
                best_solution = NestSolution(
                    placements=[
                        {
                            "uid": p.uid,
                            "source": p.source,
                            "x": p.x,
                            "y": p.y,
                            "rotation": p.rotation,
                            "sheet_uid": p.sheet_uid,
                            "polygons": p.polygons,
                            "flip_h": p.flip_h,
                            "flip_v": p.flip_v,
                        }
                        for p in identity_result.placements
                    ],
                    fitness=identity_result.fitness,
                    area_used=identity_result.area_used,
                )
        else:
            best_solution = None

        generations_without_improvement = 0

        for generation in range(num_generations):
            if self._cancelled:
                logger.info("Nesting cancelled at generation %d", generation)
                break

            # Check target from previous generation/identity check
            if (
                best_solution
                and target_fitness > 0
                and best_solution.fitness <= target_fitness
            ):
                logger.info(
                    "Early termination: Target utilization reached (%.2f%%)",
                    (1.0 / best_solution.fitness) * 100,
                )
                break

            generation_improved = False
            for individual in self._ga.population:
                if individual.fitness is not None or individual.processing:
                    continue

                individual.processing = True
                result = place_parts(
                    parts,
                    sheets,
                    individual.rotation,
                    self.config,
                    flips_h=individual.flip_h,
                    flips_v=individual.flip_v,
                )

                if result:
                    individual.fitness = result.fitness
                else:
                    individual.fitness = float("inf")
                individual.processing = False

                if result:
                    # Only calculate bounding box when we have a candidate
                    # that's potentially better than best
                    solution = NestSolution(
                        placements=[
                            {
                                "uid": p.uid,
                                "source": p.source,
                                "x": p.x,
                                "y": p.y,
                                "rotation": p.rotation,
                                "sheet_uid": p.sheet_uid,
                                "polygons": p.polygons,
                                "flip_h": p.flip_h,
                                "flip_v": p.flip_v,
                            }
                            for p in result.placements
                        ],
                        fitness=result.fitness,
                        area_used=result.area_used,
                    )

                    if (
                        best_solution is None
                        or solution.fitness < best_solution.fitness
                    ):
                        best_solution = solution
                        generation_improved = True

                        # Immediate check after finding improvement
                        if (
                            target_fitness > 0
                            and best_solution.fitness <= target_fitness
                        ):
                            logger.info(
                                "Early termination: Target utilization "
                                "reached (%.2f%%)",
                                (1.0 / best_solution.fitness) * 100,
                            )
                            break

            # If we broke inner loop due to target reached
            if (
                best_solution
                and target_fitness > 0
                and best_solution.fitness <= target_fitness
            ):
                break

            if not generation_improved:
                generations_without_improvement += 1
            else:
                generations_without_improvement = 0

            if not self._cancelled:
                self._ga.generation()

            if generation % 20 == 0 or generation == num_generations - 1:
                if best_solution:
                    logger.debug(
                        "Generation %d/%d: best fitness=%.4f",
                        generation + 1,
                        num_generations,
                        best_solution.fitness,
                    )

            if generations_without_improvement >= 5:
                logger.info(
                    "Early termination: no improvement for %d generations",
                    generations_without_improvement,
                )
                break

        return best_solution

    async def async_nest(
        self,
        task_manager: "TaskManager",
        context: "Optional[ExecutionContext]" = None,
        num_generations: Optional[int] = None,
        generations_without_improvement_limit: int = 5,
        max_parallel_tasks: int = 4,
    ) -> Optional[NestSolution]:
        clear_nfp_cache()

        if not self._workpieces:
            logger.warning("async_nest: no workpieces to nest")
            return None

        if not self._sheets:
            default_sheet = self._create_default_sheet()
            if default_sheet is None:
                logger.warning("async_nest: failed to create default sheet")
                return None
            self._sheets.append(default_sheet)
            logger.debug("async_nest: created default sheet")

        parts = self._prepare_parts()
        if not parts:
            logger.warning("async_nest: no parts after preparation")
            return None

        num_parts = len(parts)
        if num_generations is None:
            if num_parts > 100:
                num_generations = 3
            elif num_parts > 50:
                num_generations = 5
            elif num_parts > 10:
                num_generations = 15
            else:
                num_generations = min(30, max(10, num_parts * 2))

        sheets = [
            SheetInfo(
                uid=s.uid,
                polygon=s.polygons[0] if len(s.polygons) > 0 else np.array([]),
                world_offset_x=s.offset_x,
                world_offset_y=s.offset_y,
            )
            for s in self._sheets
        ]

        logger.info(
            "Starting async nest: %d part(s), %d generation(s), pop=%d, "
            "parallel=%d",
            num_parts,
            num_generations,
            self.config.population_size,
            max_parallel_tasks,
        )

        ga = GeneticAlgorithm(parts, self.config)
        state = _AsyncNestingState()

        target_fitness = 0.0
        if self.config.target_utilization > 0:
            target_fitness = 1.0 / self.config.target_utilization

        def on_individual_done(task):
            key = task.key
            with state.lock:
                if key not in state.pending_tasks:
                    return
                idx = state.pending_tasks.pop(key)

                if task.get_status() != "completed":
                    return

                result = task.result()
                state.results.append((key, idx, result))

        def process_pending_results():
            while state.results:
                key, idx, result = state.results.pop(0)
                if result and result.fitness is not None:
                    fitness = result.fitness
                    ga.population[idx].fitness = fitness
                    ga.population[idx].processing = False

                    if fitness < state.best_fitness:
                        state.best_fitness = fitness
                        state.best_placements = [
                            Placement(
                                id=p.id,
                                source=p.source,
                                uid=p.uid,
                                x=p.x,
                                y=p.y,
                                rotation=p.rotation,
                                polygons=p.polygons,
                                hulls=p.hulls,
                                sheet_uid=p.sheet_uid,
                                flip_h=p.flip_h,
                                flip_v=p.flip_v,
                            )
                            for p in result.placements
                        ]
                        state.improved_this_generation = True

                        # Check target
                        if (
                            target_fitness > 0
                            and state.best_fitness <= target_fitness
                        ):
                            logger.info(
                                "Early termination: Target utilization "
                                "reached (%.2f%%)",
                                (1.0 / state.best_fitness) * 100,
                            )
                            state.done = True

        while state.generation < num_generations and not state.done:
            state.improved_this_generation = False

            # Spawn tasks, respecting max_parallel_tasks
            for idx, ind in enumerate(ga.population):
                with state.lock:
                    if len(state.pending_tasks) >= max_parallel_tasks:
                        break
                    if state.done:
                        break

                if ind.fitness is None and not ind.processing:
                    ind.processing = True
                    # Avoid deepcopy to save memory and GC churn.
                    task_key = f"nest-eval-{state.generation}-{idx}"

                    with state.lock:
                        state.pending_tasks[task_key] = idx

                    task_manager.run_process(
                        _place_parts_worker,
                        parts,
                        sheets,
                        ind.rotation,
                        self.config,
                        flips_h=ind.flip_h,
                        flips_v=ind.flip_v,
                        key=task_key,
                        when_done=on_individual_done,
                        visible=False,
                    )

            if state.done:
                break

            await asyncio.sleep(0.05)

            with state.lock:
                process_pending_results()

                # Check if entire generation is complete
                active_count = len(state.pending_tasks)

            # Report progress
            if context:
                processed_count = sum(
                    1 for ind in ga.population if ind.fitness is not None
                )
                total_individuals = len(ga.population)
                generation_progress = (
                    processed_count / total_individuals
                    if total_individuals > 0
                    else 0
                )
                overall_progress = (
                    state.generation + generation_progress
                ) / num_generations
                context.set_progress(0.2 + 0.8 * overall_progress)
                context.set_message(
                    f"Nesting: Generation {state.generation + 1}/"
                    f"{num_generations}"
                )

            if state.done:
                break

            # If we have no active tasks and everyone is processed, move to
            # next gen
            all_processed = all(
                ind.fitness is not None for ind in ga.population
            )

            if active_count == 0 and all_processed:
                ga.generation()

                if state.improved_this_generation:
                    state.generations_without_improvement = 0
                else:
                    state.generations_without_improvement += 1

                if state.generation % 5 == 0 and state.best_fitness < float(
                    "inf"
                ):
                    logger.debug(
                        "Generation %d/%d: best fitness=%.4f",
                        state.generation + 1,
                        num_generations,
                        state.best_fitness,
                    )

                if (
                    state.generations_without_improvement
                    >= generations_without_improvement_limit
                ):
                    logger.info(
                        "Early termination: no improvement for %d gens",
                        state.generations_without_improvement,
                    )
                    state.done = True
                    break

                state.generation += 1

        for key in list(state.pending_tasks.keys()):
            task_manager.cancel_task(key)
        state.pending_tasks.clear()

        with state.lock:
            process_pending_results()

        if not state.best_placements:
            return None

        logger.debug(
            "Async nest complete: %d placement(s), fitness=%.4f",
            len(state.best_placements),
            state.best_fitness,
        )

        return NestSolution(
            placements=[
                {
                    "uid": p.uid,
                    "source": p.source,
                    "x": p.x,
                    "y": p.y,
                    "rotation": p.rotation,
                    "sheet_uid": p.sheet_uid,
                    "polygons": p.polygons,
                    "hulls": p.hulls,
                    "flip_h": p.flip_h,
                    "flip_v": p.flip_v,
                }
                for p in state.best_placements
            ],
            fitness=state.best_fitness,
            area_used=0,
        )

    def cancel(self) -> None:
        self._cancelled = True
        self._working = False

    def _prepare_parts(self) -> List[Dict[str, Any]]:
        parts = []

        for wp in self._workpieces:
            for q in range(wp.quantity):
                total_area = sum(
                    abs(polygon_area_numpy(p)) for p in wp.polygons
                )

                parts.append(
                    {
                        "id": len(parts),
                        "source": wp.source,
                        "uid": wp.uid,
                        "polygons": wp.polygons,
                        "hulls": wp.hulls,  # Pass hulls to placement
                        "area": total_area,
                    }
                )

        parts.sort(key=lambda p: p["area"], reverse=True)
        logger.debug("Prepared %d parts for nesting", len(parts))

        return parts

    def _create_default_sheet(self) -> Optional[WorkpieceInfo]:
        if not self._workpieces:
            return None

        all_bounds = []
        total_area = 0.0
        for wp in self._workpieces:
            for poly in wp.polygons:
                all_bounds.append(polygon_bounds_numpy(poly))
                total_area += abs(polygon_area_numpy(poly))

        if not all_bounds:
            return None

        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)

        parts_width = max_x - min_x
        parts_height = max_y - min_y

        estimated_width = math.sqrt(total_area * 1.5) + 10
        estimated_height = math.sqrt(total_area * 1.5) + 10

        sheet_width = max(parts_width + 10, estimated_width)
        sheet_height = max(parts_height + 10, estimated_height)

        sheet_polygon = np.array(
            [
                [0, 0],
                [sheet_width, 0],
                [sheet_width, sheet_height],
                [0, sheet_height],
            ]
        )

        logger.debug(
            "Created default sheet: size=%.2f x %.2f, parts_area=%.2f",
            sheet_width,
            sheet_height,
            total_area,
        )

        return WorkpieceInfo(
            uid="default_sheet",
            polygons=[sheet_polygon],
            source=-1,
            quantity=1,
            is_sheet=True,
            hulls=[np.array(convex_hull(sheet_polygon.tolist()))],
        )
