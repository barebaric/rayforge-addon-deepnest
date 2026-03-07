"""
Backend entry point for deepnest addon.

Registers layout strategies with the main application.
"""

import gettext
import logging
from pathlib import Path

from rayforge.core.hooks import hookimpl
from .deepnest.models import NestConfig
from .nesting import NestingLayoutStrategy

_localedir = Path(__file__).parent.parent / "locales"
_t = gettext.translation("deepnest", localedir=_localedir, fallback=True)
_ = _t.gettext

ADDON_NAME = "deepnest"

logger = logging.getLogger(__name__)


def execute_nesting(editor, items, config: NestConfig):
    """Execute nesting layout with the given configuration."""
    logger.debug(
        "execute_nesting called with config: spacing=%.3f, merge_lines=%s",
        config.spacing,
        config.merge_lines,
    )
    strategy = NestingLayoutStrategy(items=items, config=config)
    editor.layout.execute_layout(strategy, _("Nesting Layout"), use_async=True)


@hookimpl
def register_layout_strategies(layout_registry):
    """Register layout strategies with the layout registry."""
    layout_registry.register(
        NestingLayoutStrategy,
        name="nesting",
        action_id="layout-nesting",
        label=_("Nesting Layout"),
        shortcut="<Ctrl><Alt>n",
        addon_name=ADDON_NAME,
    )
