from deepnest.deepnest.spatial_grid import SpatialGrid


class TestSpatialGrid:
    def test_init_default_cell_size(self):
        grid = SpatialGrid()
        assert grid.cell_size == 50.0
        assert grid.cells == {}

    def test_init_custom_cell_size(self):
        grid = SpatialGrid(cell_size=100.0)
        assert grid.cell_size == 100.0

    def test_clear(self):
        grid = SpatialGrid()
        grid.cells[(0, 0)] = [1, 2]
        grid.clear()
        assert grid.cells == {}

    def test_get_cell(self):
        grid = SpatialGrid(cell_size=10.0)
        assert grid._get_cell(5.0, 5.0) == (0, 0)
        assert grid._get_cell(15.0, 25.0) == (1, 2)
        assert grid._get_cell(-5.0, -15.0) == (-1, -2)

    def test_get_cells_for_bbox_single_cell(self):
        grid = SpatialGrid(cell_size=10.0)
        bbox = (1.0, 1.0, 5.0, 5.0)
        cells = grid._get_cells_for_bbox(bbox)
        assert cells == {(0, 0)}

    def test_get_cells_for_bbox_multiple_cells(self):
        grid = SpatialGrid(cell_size=10.0)
        bbox = (5.0, 5.0, 25.0, 25.0)
        cells = grid._get_cells_for_bbox(bbox)
        expected = {
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        }
        assert cells == expected

    def test_insert_single_bbox(self):
        grid = SpatialGrid(cell_size=10.0)
        bbox = (5.0, 5.0, 15.0, 15.0)
        grid.insert(0, bbox)
        assert (0, 0) in grid.cells
        assert (1, 0) in grid.cells
        assert (0, 1) in grid.cells
        assert (1, 1) in grid.cells
        assert 0 in grid.cells[(0, 0)]

    def test_insert_multiple_bboxes(self):
        grid = SpatialGrid(cell_size=10.0)
        grid.insert(0, (0.0, 0.0, 5.0, 5.0))
        grid.insert(1, (0.0, 0.0, 5.0, 5.0))
        assert grid.cells[(0, 0)] == [0, 1]

    def test_query_empty_grid(self):
        grid = SpatialGrid()
        result = grid.query((0.0, 0.0, 10.0, 10.0))
        assert result == set()

    def test_query_no_match(self):
        grid = SpatialGrid(cell_size=10.0)
        grid.insert(0, (100.0, 100.0, 110.0, 110.0))
        result = grid.query((0.0, 0.0, 10.0, 10.0))
        assert result == set()

    def test_query_single_match(self):
        grid = SpatialGrid(cell_size=10.0)
        grid.insert(0, (5.0, 5.0, 15.0, 15.0))
        result = grid.query((0.0, 0.0, 10.0, 10.0))
        assert result == {0}

    def test_query_multiple_matches(self):
        grid = SpatialGrid(cell_size=10.0)
        grid.insert(0, (0.0, 0.0, 10.0, 10.0))
        grid.insert(1, (5.0, 5.0, 15.0, 15.0))
        grid.insert(2, (20.0, 20.0, 30.0, 30.0))
        result = grid.query((0.0, 0.0, 10.0, 10.0))
        assert result == {0, 1}

    def test_query_returns_set(self):
        grid = SpatialGrid(cell_size=10.0)
        grid.insert(0, (0.0, 0.0, 10.0, 10.0))
        result = grid.query((0.0, 0.0, 10.0, 10.0))
        assert isinstance(result, set)

    def test_negative_coordinates(self):
        grid = SpatialGrid(cell_size=10.0)
        grid.insert(0, (-15.0, -15.0, -5.0, -5.0))
        result = grid.query((-12.0, -12.0, -8.0, -8.0))
        assert result == {0}
