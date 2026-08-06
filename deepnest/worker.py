"""
Backend entry point for deepnest addon.

Registers layout strategies with the main application.
"""

import logging
from gettext import gettext as _

from rayforge.core.hooks import hookimpl

from .deepnest.models import NestConfig
from .nesting import NestingLayoutStrategy

ADDON_NAME = "deepnest"

logger = logging.getLogger(__name__)


def execute_nesting(editor, items, config: NestConfig):
    """Execute nesting layout with the given configuration."""
    logger.debug(
        "execute_nesting called with config: spacing=%.3f, rotations=%d",
        config.spacing,
        config.rotations,
    )
    strategy = NestingLayoutStrategy(items=items, config=config)
    editor.layout.execute_layout(strategy, _("Nesting Layout"), use_async=True)


@hookimpl
def register_layout_strategies(layout_registry):
    """Register layout strategies with the layout registry."""
    layout_registry.register(
        NestingLayoutStrategy,
        name="nesting",
        addon_name=ADDON_NAME,
    )
