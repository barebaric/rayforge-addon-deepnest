"""
Backend entry point for deepnest addon.

Registers layout strategies and actions with the main application.
"""

import gettext
from pathlib import Path

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from .nesting import NestingLayoutStrategy

_localedir = Path(__file__).parent.parent / "locales"
_t = gettext.translation("deepnest", localedir=_localedir, fallback=True)
_ = _t.gettext

ADDON_NAME = "deepnest"


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


@hookimpl
def register_actions(window):
    """Register window actions."""
    action = Gio.SimpleAction.new("layout-nesting", None)

    def on_activate(action, param):
        editor = window.doc_editor
        items = list(window.surface.get_selected_items())
        items_to_layout = editor.layout.get_items_to_layout(items)

        if not items_to_layout:
            return

        strategy = NestingLayoutStrategy(
            items=items_to_layout,
            spacing=0.0,
            rotations=36,
            population_size=10,
            merge_lines=True,
        )
        editor.layout.execute_layout(
            strategy, _("Nesting Layout"), use_async=True
        )

    action.connect("activate", on_activate)
    window.add_action(action)
