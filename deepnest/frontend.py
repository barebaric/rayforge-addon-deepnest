"""
Frontend entry point for deepnest addon.

Registers menu items and actions with the main application.
"""

import gettext
import logging
from pathlib import Path
from typing import Optional

from gi.repository import Gio

from rayforge.context import get_context
from rayforge.core.hooks import hookimpl
from .backend import execute_nesting
from .deepnest.models import NestConfig
from .dialog import NestingSettingsDialog

_localedir = Path(__file__).parent.parent / "locales"
_t = gettext.translation("deepnest", localedir=_localedir, fallback=True)
_ = _t.gettext

ADDON_NAME = "deepnest"

logger = logging.getLogger(__name__)

_session_config: Optional[NestConfig] = None


@hookimpl
def register_menu_items(menu_registry):
    """Register menu items with the menu registry."""
    menu_registry.register(
        item_id="deepnest.layout_nesting",
        label=_("Auto Layout (Nesting)"),
        action="win.layout-nesting",
        menu="Arrange",
        priority=50,
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

        machine = get_context().machine
        if machine and machine.heads:
            default_spacing = machine.heads[0].spot_size_mm[0]
        else:
            default_spacing = 0.1

        if _session_config is not None:
            initial_spacing = _session_config.spacing
            initial_merge_lines = _session_config.merge_lines
        else:
            initial_spacing = default_spacing
            initial_merge_lines = True

        dialog = NestingSettingsDialog(
            window,
            initial_spacing=initial_spacing,
            initial_merge_lines=initial_merge_lines,
        )

        def on_dialog_response(dialog, response_id):
            global _session_config

            if response_id == "start":
                config = dialog.get_config()
                _session_config = config
                logger.debug(
                    "Saved session config: spacing=%.3f, merge_lines=%s",
                    config.spacing,
                    config.merge_lines,
                )
                execute_nesting(editor, items_to_layout, config)
            dialog.close()

        dialog.connect("response", on_dialog_response)
        dialog.present()

    action.connect("activate", on_activate)
    window.add_action(action)
