"""
Implements a nesting layout strategy using the deepnest module.
"""

from __future__ import annotations

import logging
import math
import os
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

from rayforge.context import get_context
from rayforge.core.geo import Geometry
from rayforge.core.group import Group
from rayforge.core.item import DocItem
from rayforge.core.matrix import Matrix
from rayforge.core.stock import StockItem
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.machine import Origin
from rayforge.doceditor.layout.base import LayoutStrategy
from .deepnest.core import DeepNest
from .deepnest.models import (
    NestConfig,
    NestSolution,
    Placement,
)
from .deepnest.placement import validate_placements_no_overlap

if TYPE_CHECKING:
    from rayforge.shared.tasker.context import ExecutionContext
    from rayforge.shared.tasker.manager import TaskManager


logger = logging.getLogger(__name__)


class NestingLayoutStrategy(LayoutStrategy):
    """
    Arranges workpieces using a genetic algorithm for optimal nesting.

    This strategy uses the deepnest module to find an efficient placement
    of workpieces on the available stock or work area in a unified world space.
    """

    def __init__(
        self,
        items: Sequence[DocItem],
        config: Optional[NestConfig] = None,
        **kwargs,
    ):
        super().__init__(items, **kwargs)
        self.config = config or NestConfig()
        self.unplaced_items: list[DocItem] = []
        logger.debug(
            "NestingLayoutStrategy created with config: spacing=%.3f, "
            "merge_lines=%s",
            self.config.spacing,
            self.config.merge_lines,
        )

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        if not self.items:
            return {}

        logger.info("Starting nesting layout...")

        if context:
            context.set_message("Preparing workpieces...")

        workpieces = self._collect_workpieces()
        if not workpieces:
            return {}

        num_workpieces = len(workpieces)
        logger.debug("Found %d workpiece(s) to nest", num_workpieces)

        initial_area = self._calculate_bounding_box_area(workpieces)
        initially_on_stock = self._get_workpieces_on_stock(workpieces)
        logger.debug(
            "Initial bounding box area: %s, workpieces on stock: %d",
            initial_area,
            len(initially_on_stock),
        )

        if context:
            context.set_progress(0.1)

        population_size = min(
            self.config.population_size, max(4, num_workpieces)
        )
        config = NestConfig(
            spacing=self.config.spacing,
            rotations=self.config.rotations,
            population_size=population_size,
            simplify_tolerance=self.config.simplify_tolerance,
            merge_lines=self.config.merge_lines,
        )

        logger.debug(
            "Creating DeepNest with config: spacing=%.3f, merge_lines=%s",
            config.spacing,
            config.merge_lines,
        )
        nester = DeepNest(config)

        for wp in workpieces:
            geo = wp.get_world_geometry()
            if geo is None:
                logger.warning("Workpiece '%s' has no world geometry", wp.uid)
                continue
            if geo.is_empty():
                logger.warning("Workpiece '%s' has empty geometry", wp.uid)
                continue
            nester.add_geometry(geo, uid=wp.uid)

        stock_polygons = self._get_stock_polygons()
        if stock_polygons:
            for stock_geo, stock_uid in stock_polygons:
                nester.add_geometry(stock_geo, uid=stock_uid, is_sheet=True)
            logger.debug("Using %d stock sheet(s)", len(stock_polygons))
        else:
            logger.debug("No stock defined; using auto-generated sheet")

        if context:
            context.set_message("Running nesting algorithm...")
            context.set_progress(0.2)

        solution = nester.nest()

        if context:
            context.set_progress(0.9)

        if not solution:
            logger.warning("Nesting failed to find a solution.")
            self.unplaced_items = list(workpieces)
            return {}

        if not self._is_solution_better_than_initial(
            solution, workpieces, initial_area, initially_on_stock
        ):
            self.unplaced_items = list(workpieces)
            return {}

        logger.debug(
            "Nesting complete: %d placement(s), fitness=%.4f",
            len(solution.placements),
            solution.fitness,
        )

        placements = self._extract_placements(solution)
        deltas = self._compute_deltas_from_placements(placements, workpieces)

        placed_uids = {p.uid for p in placements}
        self.unplaced_items = [
            wp for wp in workpieces if wp.uid not in placed_uids
        ]

        if self.unplaced_items:
            logger.warning(
                "%d workpiece(s) could not be placed",
                len(self.unplaced_items),
            )
            self._handle_unplaced_items(deltas)

        logger.info("Nesting layout complete.")
        return deltas

    async def calculate_deltas_async(
        self,
        context: Optional[ExecutionContext] = None,
        task_manager: "Optional[TaskManager]" = None,
    ) -> Dict[DocItem, Matrix]:
        if task_manager is None:
            raise ValueError("task_manager is required for async nesting")
        if not self.items:
            return {}

        logger.info("Starting nesting layout (parallel)...")

        workpieces = self._collect_workpieces()
        if not workpieces:
            return {}

        num_workpieces = len(workpieces)
        logger.debug("Found %d workpiece(s) to nest", num_workpieces)

        initial_area = self._calculate_bounding_box_area(workpieces)
        initially_on_stock = self._get_workpieces_on_stock(workpieces)
        logger.debug(
            "Initial bounding box area: %s, workpieces on stock: %d",
            initial_area,
            len(initially_on_stock),
        )

        cpu_count = os.cpu_count() or 4
        max_workers = max(1, int(cpu_count * 0.9))

        population_size = min(max_workers, max(8, int(num_workpieces * 0.5)))

        config = NestConfig(
            spacing=self.config.spacing,
            rotations=self.config.rotations,
            population_size=population_size,
            simplify_tolerance=self.config.simplify_tolerance,
            merge_lines=self.config.merge_lines,
        )

        logger.debug(
            "Creating DeepNest with config: spacing=%.3f, merge_lines=%s",
            config.spacing,
            config.merge_lines,
        )
        nester = DeepNest(config)

        for wp in workpieces:
            geo = wp.get_world_geometry()
            if geo is None:
                logger.warning(
                    "Workpiece '%s' has no world geometry - source_segment=%s",
                    wp.uid,
                    wp.source_segment is not None,
                )
                continue
            if geo.is_empty():
                logger.warning("Workpiece '%s' has empty geometry", wp.uid)
                continue
            nester.add_geometry(geo, uid=wp.uid)

        stock_polygons = self._get_stock_polygons()
        if stock_polygons:
            for stock_geo, stock_uid in stock_polygons:
                nester.add_geometry(stock_geo, uid=stock_uid, is_sheet=True)
            logger.debug("Using %d stock sheet(s)", len(stock_polygons))
        else:
            logger.debug("No stock defined; using auto-generated sheet")

        if context:
            context.set_message("Running nesting algorithm...")
            context.set_progress(0.2)

        solution = await nester.async_nest(
            task_manager, max_parallel_tasks=max_workers
        )

        if not solution:
            logger.warning("Nesting found no solution")
            self.unplaced_items = list(workpieces)
            return {}

        if not self._is_solution_better_than_initial(
            solution, workpieces, initial_area, initially_on_stock
        ):
            self.unplaced_items = list(workpieces)
            return {}

        logger.debug(
            "Nesting complete: %d placement(s), fitness=%.4f",
            len(solution.placements),
            solution.fitness,
        )

        placements = self._extract_placements(solution)
        deltas = self._compute_deltas_from_placements(placements, workpieces)

        placed_uids = {p.uid for p in placements}
        self.unplaced_items = [
            wp for wp in workpieces if wp.uid not in placed_uids
        ]

        if self.unplaced_items:
            logger.warning(
                "%d workpiece(s) could not be placed",
                len(self.unplaced_items),
            )
            self._handle_unplaced_items(deltas)

        logger.info("Nesting layout complete.")
        return deltas

    def _extract_placements(self, solution: NestSolution) -> List[Placement]:
        placements = []
        for p in solution.placements:
            placements.append(
                Placement(
                    id=p.get("id", 0),
                    source=p.get("source", 0),
                    uid=p["uid"],
                    x=p["x"],
                    y=p["y"],
                    rotation=p["rotation"],
                    polygons=p["polygons"],
                    sheet_uid=p.get("sheet_uid"),
                )
            )
        return placements

    def _collect_workpieces(self) -> list[WorkPiece]:
        workpieces = []
        for item in self.items:
            if isinstance(item, WorkPiece):
                workpieces.append(item)
            elif isinstance(item, Group):
                workpieces.extend(item.get_descendants(of_type=WorkPiece))
        return workpieces

    def _calculate_bounding_box_area(
        self, workpieces: list[WorkPiece]
    ) -> Optional[float]:
        valid_bboxes = []
        for wp in workpieces:
            geo = wp.get_world_geometry()
            if geo is None or geo.is_empty():
                continue
            min_x, min_y, max_x, max_y = geo.rect()
            valid_bboxes.append((min_x, min_y, max_x, max_y))

        if not valid_bboxes:
            return None

        all_min_x = min(b[0] for b in valid_bboxes)
        all_min_y = min(b[1] for b in valid_bboxes)
        all_max_x = max(b[2] for b in valid_bboxes)
        all_max_y = max(b[3] for b in valid_bboxes)

        return (all_max_x - all_min_x) * (all_max_y - all_min_y)

    def _get_workpieces_on_stock(
        self, workpieces: list[WorkPiece]
    ) -> set[WorkPiece]:
        stock_polygons = self._get_stock_polygons()
        if not stock_polygons:
            return set(workpieces)

        on_stock = set()
        for wp in workpieces:
            geo = wp.get_world_geometry()
            if geo is None or geo.is_empty():
                continue
            wp_min_x, wp_min_y, wp_max_x, wp_max_y = geo.rect()
            for stock_geo, _ in stock_polygons:
                s_min_x, s_min_y, s_max_x, s_max_y = stock_geo.rect()
                if (
                    wp_min_x >= s_min_x
                    and wp_min_y >= s_min_y
                    and wp_max_x <= s_max_x
                    and wp_max_y <= s_max_y
                ):
                    on_stock.add(wp)
                    break

        return on_stock

    def _is_solution_better_than_initial(
        self,
        solution: NestSolution,
        workpieces: list[WorkPiece],
        initial_area: Optional[float],
        initially_on_stock: set[WorkPiece],
    ) -> bool:
        placed_uids = {p.get("uid") for p in solution.placements}
        now_placed = {wp for wp in workpieces if wp.uid in placed_uids}

        for wp in initially_on_stock:
            if wp not in now_placed:
                logger.warning(
                    "Rejecting nesting solution: workpiece '%s' was on stock "
                    "but is now unplaced",
                    wp.uid,
                )
                return False

        return True

    def _calculate_placements_bounding_box_area(
        self, solution: NestSolution
    ) -> Optional[float]:
        if not solution.placements:
            return None

        all_points = []
        for placement in solution.placements:
            x = placement.get("x", 0)
            y = placement.get("y", 0)
            rotation = placement.get("rotation", 0)
            polygons = placement.get("polygons", [])

            cos_r = math.cos(math.radians(rotation))
            sin_r = math.sin(math.radians(rotation))

            for poly in polygons:
                if poly is None or len(poly) == 0:
                    continue
                for point in poly:
                    px, py = point[0], point[1]
                    rx = px * cos_r - py * sin_r + x
                    ry = px * sin_r + py * cos_r + y
                    all_points.append((rx, ry))

        if not all_points:
            return None

        min_x = min(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_x = max(p[0] for p in all_points)
        max_y = max(p[1] for p in all_points)

        return (max_x - min_x) * (max_y - min_y)

    def _get_stock_geometry(self, stock_item: StockItem) -> Optional[Geometry]:
        """
        Returns the stock's world-space geometry exactly.
        """
        geo = stock_item.get_world_geometry()
        if geo is None or geo.is_empty():
            geo = stock_item.get_world_rect_geometry()

        if geo is None or geo.is_empty():
            return None

        return geo

    def _get_stock_polygons(self) -> List[tuple[Geometry, str]]:
        """
        Returns a list of (geometry, uid) tuples for all stock items.
        """
        selected_stocks = [
            item for item in self.items if isinstance(item, StockItem)
        ]

        if selected_stocks:
            stocks = []
            for idx, stock_item in enumerate(selected_stocks):
                geo = self._get_stock_geometry(stock_item)
                if geo and not geo.is_empty():
                    uid = f"stock-{idx}"
                    stocks.append((geo, uid))
            if stocks:
                return stocks

        doc = self.items[0].doc if self.items else None
        stocks = []

        if doc:
            for idx, stock_item in enumerate(doc.stock_items):
                geo = self._get_stock_geometry(stock_item)
                if geo and not geo.is_empty():
                    uid = f"stock-{idx}"
                    stocks.append((geo, uid))

        if stocks:
            return stocks

        machine = get_context().machine
        if machine:
            ref_x, ref_y = machine.get_workarea_origin_offset()
            _, _, wa_w, wa_h = machine.work_area
            origin = machine.origin

            if origin == Origin.BOTTOM_LEFT:
                sheet_x, sheet_y = ref_x, ref_y
            elif origin == Origin.TOP_LEFT:
                sheet_x, sheet_y = ref_x, ref_y - wa_h
            elif origin == Origin.BOTTOM_RIGHT:
                sheet_x, sheet_y = ref_x - wa_w, ref_y
            else:
                sheet_x, sheet_y = ref_x - wa_w, ref_y - wa_h

            geometry = Geometry()
            geometry.move_to(sheet_x, sheet_y)
            geometry.line_to(sheet_x + wa_w, sheet_y)
            geometry.line_to(sheet_x + wa_w, sheet_y + wa_h)
            geometry.line_to(sheet_x, sheet_y + wa_h)
            geometry.close_path()
            uid = "machine-workarea"
            return [(geometry, uid)]

        return []

    def _compute_deltas_from_placements(
        self,
        placements: List[Placement],
        workpieces: List[WorkPiece],
    ) -> Dict[DocItem, Matrix]:
        deltas: Dict[DocItem, Matrix] = {}
        workpiece_map = {wp.uid: wp for wp in workpieces}

        for placement in placements:
            uid = placement.uid
            if uid not in workpiece_map:
                logger.warning("Placement has unknown uid: %s", uid)
                continue

            wp = workpiece_map[uid]
            old_world = wp.get_world_transform()
            world_geo = wp.get_world_geometry()

            if world_geo is None or world_geo.is_empty():
                continue

            # 1. Translate the original world shape so its
            #    min_x/min_y is at (0,0)
            # This replicates Deepnest's normalize_polygons()
            min_x, min_y, _, _ = world_geo.rect()
            T_to_origin = Matrix.translation(-min_x, -min_y)

            # 2. Delta Rotation from the Genetic Algorithm
            R = Matrix.rotation(placement.rotation)

            # 3. Apply steps 1 & 2 to find the new bounding box's
            #    min corner.
            # Deepnest re-normalizes the rotated shape so its
            # minimum lies at the exact coordinates of placement.x
            # and placement.y on the sheet.
            temp_geo = world_geo.copy()
            temp_geo.transform((R @ T_to_origin).to_4x4_numpy())
            rot_min_x, rot_min_y, _, _ = temp_geo.rect()

            # 4. Translate so the minimum corner lands on the target placement
            T_to_placement = Matrix.translation(
                placement.x - rot_min_x, placement.y - rot_min_y
            )

            # 5. Compose the full affine transformation mapping the original
            # world shape to the final nested world shape.
            M_world_delta = T_to_placement @ R @ T_to_origin

            # 6. Apply to the old world matrix
            new_world = M_world_delta @ old_world

            # 7. Convert the new world matrix back into local delta
            old_local = wp.matrix
            if old_local.has_zero_scale():
                continue

            old_local_inv = old_local.invert()
            parent_inv = Matrix.identity()
            if wp.parent:
                parent_world = wp.parent.get_world_transform()
                if not parent_world.has_zero_scale():
                    parent_inv = parent_world.invert()

            delta = parent_inv @ new_world @ old_local_inv
            deltas[wp] = delta

        if len(placements) > 1:
            validate_placements_no_overlap(placements)

        return deltas

    def _handle_unplaced_items(self, deltas: Dict[DocItem, Matrix]) -> None:
        sheets = self._get_stock_polygons()
        if not sheets:
            return

        sheet_right = float("-inf")
        sheet_top = float("inf")
        for geo, _ in sheets:
            min_x, min_y, max_x, max_y = geo.rect()
            sheet_right = max(sheet_right, max_x)
            sheet_top = min(sheet_top, min_y)

        if sheet_right == float("-inf"):
            return

        unplaced_bboxes = []
        for item in self.unplaced_items:
            get_bbox = getattr(item, "get_geometry_world_bbox", None)
            if get_bbox:
                bbox = get_bbox()
                if bbox:
                    unplaced_bboxes.append(bbox)
            else:
                x, y, w, h = item.bbox
                unplaced_bboxes.append((x, y, x + w, y + h))

        valid_bboxes = [b for b in unplaced_bboxes if b]
        if not valid_bboxes:
            return

        min_x = min(b[0] for b in valid_bboxes)
        min_y = min(b[1] for b in valid_bboxes)

        target_x = sheet_right + self.config.spacing * 4
        target_y = sheet_top

        dx = target_x - min_x
        dy = target_y - min_y

        for item in self.unplaced_items:
            old_world = item.get_world_transform()
            final_matrix = Matrix.translation(dx, dy) @ old_world

            old_local = item.matrix
            if old_local.has_zero_scale():
                continue

            old_local_inv = old_local.invert()
            parent_inv = Matrix.identity()
            if item.parent:
                parent_world = item.parent.get_world_transform()
                if not parent_world.has_zero_scale():
                    parent_inv = parent_world.invert()

            delta = parent_inv @ final_matrix @ old_local_inv
            deltas[item] = delta
