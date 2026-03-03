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
    of workpieces on the available stock or work area.
    """

    def __init__(
        self,
        items: Sequence[DocItem],
        spacing: float = 0.0,
        rotations: int = 36,
        population_size: int = 10,
        **kwargs,
    ):
        super().__init__(items, **kwargs)
        self.spacing = spacing
        self.rotations = rotations
        self.population_size = population_size
        self.unplaced_items: list[DocItem] = []

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

        if num_workpieces < 2:
            logger.info(
                "Nesting requires at least 2 workpieces; skipping layout."
            )
            return {}

        if context:
            context.set_progress(0.1)

        population_size = min(self.population_size, max(4, num_workpieces))
        config = NestConfig(
            spacing=self.spacing,
            rotations=self.rotations,
            population_size=population_size,
        )

        nester = DeepNest(config)

        for wp in workpieces:
            geo = wp.get_world_geometry()
            if geo and not geo.is_empty():
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

        logger.debug(
            "Nesting complete: %d placement(s), fitness=%.4f",
            len(solution.placements),
            solution.fitness,
        )

        deltas = self._compute_deltas_from_solution(solution)

        placed_uids = {p["uid"] for p in solution.placements}
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

        if num_workpieces < 2:
            logger.info(
                "Nesting requires at least 2 workpieces; skipping layout."
            )
            return {}

        cpu_count = os.cpu_count() or 4
        max_workers = int(cpu_count * 0.9)
        population_size = min(max_workers, max(8, num_workpieces * 2))

        config = NestConfig(
            spacing=self.spacing,
            rotations=self.rotations,
            population_size=population_size,
        )

        nester = DeepNest(config)

        for wp in workpieces:
            geo = wp.get_world_geometry()
            if geo and not geo.is_empty():
                nester.add_geometry(geo, uid=wp.uid)

        stock_polygons = self._get_stock_polygons()
        if stock_polygons:
            for stock_geo, stock_uid in stock_polygons:
                nester.add_geometry(stock_geo, uid=stock_uid, is_sheet=True)
            logger.debug("Using %d stock sheet(s)", len(stock_polygons))
        else:
            logger.debug("No stock defined; using auto-generated sheet")

        solution = await nester.async_nest(task_manager)

        if not solution:
            logger.warning("Nesting found no solution")
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

    def _get_stock_polygons(self) -> List[tuple[Geometry, str]]:
        """
        Returns a list of (geometry, uid) tuples for all stock items.
        """
        doc = self.items[0].doc if self.items else None
        stocks = []

        if doc:
            for idx, stock_item in enumerate(doc.stock_items):
                if stock_item and stock_item.bbox:
                    x, y, w, h = stock_item.bbox
                    if w > 0 and h > 0:
                        geometry = Geometry()
                        geometry.move_to(x, y)
                        geometry.line_to(x + w, y)
                        geometry.line_to(x + w, y + h)
                        geometry.line_to(x, y + h)
                        geometry.close_path()
                        logger.debug(
                            "Using stock bbox as sheet %d: "
                            "(%.2f, %.2f) %.2f x %.2f",
                            idx,
                            x,
                            y,
                            w,
                            h,
                        )
                        stocks.append((geometry, f"stock-{idx}"))

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
            logger.debug(
                "Using machine work area as sheet at ref point: "
                "(%.2f, %.2f) %.2f x %.2f, origin=%s",
                sheet_x,
                sheet_y,
                wa_w,
                wa_h,
                origin.name,
            )
            return [(geometry, "machine-workarea")]

        return []

    def _compute_deltas_from_solution(
        self, solution: NestSolution
    ) -> Dict[DocItem, Matrix]:
        deltas: Dict[DocItem, Matrix] = {}

        workpiece_map = {wp.uid: wp for wp in self._collect_workpieces()}

        for placement in solution.placements:
            uid = placement.get("uid")
            if not uid or uid not in workpiece_map:
                continue

            wp = workpiece_map[uid]
            x = placement.get("x", 0)
            y = placement.get("y", 0)
            rotation = placement.get("rotation", 0)
            sheet_uid = placement.get("sheet_uid")

            if sheet_uid:
                logger.debug(
                    "Placing '%s' on sheet '%s' at (%.2f, %.2f)",
                    uid,
                    sheet_uid,
                    x,
                    y,
                )

            old_world = wp.get_world_transform()
            old_scale_w, old_scale_h = old_world.get_abs_scale()

            if old_scale_w <= 0 or old_scale_h <= 0:
                continue

            center = (old_scale_w / 2, old_scale_h / 2)
            T = Matrix.translation(x, y)
            R = Matrix.rotation(rotation, center=center)
            S = Matrix.scale(old_scale_w, old_scale_h)
            final_matrix = T @ R @ S

            old_local = wp.matrix
            if old_local.has_zero_scale():
                continue

            old_local_inv = old_local.invert()
            parent_inv = Matrix.identity()
            if wp.parent:
                parent_world = wp.parent.get_world_transform()
                if not parent_world.has_zero_scale():
                    parent_inv = parent_world.invert()

            delta = parent_inv @ final_matrix @ old_local_inv
            deltas[wp] = delta

        return deltas

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
            rotation = placement.rotation
            placement_x = placement.x
            placement_y = placement.y

            geo = wp.get_world_geometry()
            if geo is None or geo.is_empty():
                logger.warning("Workpiece '%s' has no geometry", uid)
                continue

            geo_polygons = geo.to_polygons(tolerance=0.1)
            if not geo_polygons:
                logger.warning("Workpiece '%s' has no polygons", uid)
                continue

            all_points = [p for poly in geo_polygons for p in poly]
            if not all_points:
                logger.warning("Workpiece '%s' has no points", uid)
                continue

            old_world = wp.get_world_transform()

            angle_rad = math.radians(rotation)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            rotated_points = [
                (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
                for x, y in all_points
            ]
            rotated_min_x = min(p[0] for p in rotated_points)
            rotated_min_y = min(p[1] for p in rotated_points)

            tx = placement_x - rotated_min_x
            ty = placement_y - rotated_min_y

            T = Matrix.translation(tx, ty)
            R = Matrix.rotation(rotation)
            final_matrix = T @ R @ old_world

            logger.debug(
                "Placement '%s': pos=(%.2f, %.2f) rot=%.1f, "
                "rotated_min=(%.2f, %.2f), translate=(%.2f, %.2f)",
                uid,
                placement_x,
                placement_y,
                rotation,
                rotated_min_x,
                rotated_min_y,
                tx,
                ty,
            )

            old_local = wp.matrix
            if old_local.has_zero_scale():
                continue

            old_local_inv = old_local.invert()
            parent_inv = Matrix.identity()
            if wp.parent:
                parent_world = wp.parent.get_world_transform()
                if not parent_world.has_zero_scale():
                    parent_inv = parent_world.invert()

            delta = parent_inv @ final_matrix @ old_local_inv
            deltas[wp] = delta

        if len(placements) > 1:
            validate_placements_no_overlap(placements)

        return deltas

    def _handle_unplaced_items(self, deltas: Dict[DocItem, Matrix]) -> None:
        doc = self.items[0].doc if self.items else None
        stock_item = None
        if doc:
            stock_items = doc.stock_items
            stock_item = stock_items[0] if stock_items else None

        if not stock_item or not stock_item.bbox:
            return

        stock_bbox = stock_item.bbox

        unplaced_bboxes = [
            self._get_item_world_bbox(item) for item in self.unplaced_items
        ]
        valid_bboxes = [b for b in unplaced_bboxes if b]
        if not valid_bboxes:
            return

        min_x = min(b[0] for b in valid_bboxes)
        max_y = max(b[3] for b in valid_bboxes)

        target_x = stock_bbox[0] + stock_bbox[2] + self.spacing * 4
        target_y = stock_bbox[1] + stock_bbox[3]

        dx = target_x - min_x
        dy = target_y - max_y

        for item in self.unplaced_items:
            old_world = item.get_world_transform()
            tx_old, ty_old = old_world.decompose()[:2]

            final_x = tx_old + dx
            final_y = ty_old + dy

            T = Matrix.translation(final_x, final_y)
            scale_w, scale_h = old_world.get_abs_scale()
            S = Matrix.scale(scale_w, scale_h)
            final_matrix = T @ S

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
