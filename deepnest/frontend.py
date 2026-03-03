"""
Frontend entry point for deepnest addon.

Registers menu items with the main application.
"""

import gettext
from pathlib import Path

from rayforge.core.hooks import hookimpl

_localedir = Path(__file__).parent.parent / "locales"
_t = gettext.translation("deepnest", localedir=_localedir, fallback=True)
_ = _t.gettext

ADDON_NAME = "deepnest"


@hookimpl
def register_menu_items(menu_registry):
    """Register menu items with the menu registry."""
    menu_registry.register(
        item_id="deepnest.layout_nesting",
        label=_("Nesting Layout"),
        action="win.layout-nesting",
        menu="Arrange",
        priority=50,
        addon_name=ADDON_NAME,
    )
