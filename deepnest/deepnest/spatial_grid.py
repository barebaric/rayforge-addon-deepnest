from typing import Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from rayforge.core.geo import Rect


class SpatialGrid:
    """Simple grid-based spatial index for bounding boxes."""

    def __init__(self, cell_size: float = 50.0):
        self.cell_size = cell_size
        self.cells: Dict[tuple[int, int], List[int]] = {}

    def clear(self):
        self.cells.clear()

    def _get_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def _get_cells_for_bbox(self, bbox: "Rect") -> Set[tuple[int, int]]:
        min_x, min_y, max_x, max_y = bbox
        cells = set()
        start_cx = int(min_x // self.cell_size)
        end_cx = int(max_x // self.cell_size)
        start_cy = int(min_y // self.cell_size)
        end_cy = int(max_y // self.cell_size)
        for cx in range(start_cx, end_cx + 1):
            for cy in range(start_cy, end_cy + 1):
                cells.add((cx, cy))
        return cells

    def insert(self, index: int, bbox: "Rect"):
        for cell in self._get_cells_for_bbox(bbox):
            if cell not in self.cells:
                self.cells[cell] = []
            self.cells[cell].append(index)

    def query(self, bbox: "Rect") -> Set[int]:
        result = set()
        for cell in self._get_cells_for_bbox(bbox):
            if cell in self.cells:
                result.update(self.cells[cell])
        return result
