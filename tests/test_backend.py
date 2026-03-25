"""
Tests for deepnest backend functions.
"""

from unittest.mock import MagicMock, patch

from deepnest.backend import execute_nesting
from deepnest.deepnest.models import NestConfig


class TestExecuteNesting:
    """Tests for execute_nesting helper function."""

    def test_strategy_created_with_config(self):
        """Verify strategy is created with the config."""
        editor = MagicMock()
        editor.layout = MagicMock()
        items = [MagicMock()]

        config = NestConfig(spacing=1.5)

        with patch(
            "deepnest.backend.NestingLayoutStrategy"
        ) as mock_strategy_class:
            mock_strategy = MagicMock()
            mock_strategy_class.return_value = mock_strategy

            execute_nesting(editor, items, config)

            mock_strategy_class.assert_called_once_with(
                items=items,
                config=config,
            )

    def test_execute_layout_called_with_strategy(self):
        """Verify execute_layout is called with the strategy."""
        editor = MagicMock()
        editor.layout = MagicMock()
        items = [MagicMock()]

        config = NestConfig(spacing=0.5)

        with patch(
            "deepnest.backend.NestingLayoutStrategy"
        ) as mock_strategy_class:
            mock_strategy = MagicMock()
            mock_strategy_class.return_value = mock_strategy

            execute_nesting(editor, items, config)

            editor.layout.execute_layout.assert_called_once_with(
                mock_strategy, "Nesting Layout", use_async=True
            )

    def test_execute_with_zero_spacing(self):
        """Verify execute_nesting works with zero spacing."""
        editor = MagicMock()
        editor.layout = MagicMock()
        items = []

        config = NestConfig(spacing=0.0)

        with patch(
            "deepnest.backend.NestingLayoutStrategy"
        ) as mock_strategy_class:
            mock_strategy = MagicMock()
            mock_strategy_class.return_value = mock_strategy

            execute_nesting(editor, items, config)

            mock_strategy_class.assert_called_once()
            call_kwargs = mock_strategy_class.call_args[1]
            assert call_kwargs["config"].spacing == 0.0

    def test_execute_with_flip_options(self):
        """Verify execute_nesting passes flip options correctly."""
        editor = MagicMock()
        editor.layout = MagicMock()
        items = [MagicMock()]

        config = NestConfig(spacing=1.0, flip_h=True, flip_v=True)

        with patch(
            "deepnest.backend.NestingLayoutStrategy"
        ) as mock_strategy_class:
            mock_strategy = MagicMock()
            mock_strategy_class.return_value = mock_strategy

            execute_nesting(editor, items, config)

            call_kwargs = mock_strategy_class.call_args[1]
            assert call_kwargs["config"].flip_h is True
            assert call_kwargs["config"].flip_v is True
