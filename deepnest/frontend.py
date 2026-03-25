"""
Frontend entry point for deepnest addon.

Registers actions with menu placement.
"""

import gettext
import logging
from pathlib import Path
from typing import Optional

from gi.repository import Gio

from rayforge.context import get_context
from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.action_registry import MenuPlacement, ToolbarPlacement
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
def register_actions(action_registry):
    """Register action with menu and toolbar placement."""
    action = Gio.SimpleAction.new("layout-nesting", None)

    def on_activate(action, param):
        window = action_registry.window
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
            initial_constrain_rotation = _session_config.rotations == 1
            initial_flip_h = _session_config.flip_h
            initial_flip_v = _session_config.flip_v
        else:
            initial_spacing = default_spacing
            initial_merge_lines = True
            initial_constrain_rotation = False
            initial_flip_h = False
            initial_flip_v = False

        dialog = NestingSettingsDialog(
            window,
            initial_spacing=initial_spacing,
            initial_merge_lines=initial_merge_lines,
            initial_constrain_rotation=initial_constrain_rotation,
            initial_flip_h=initial_flip_h,
            initial_flip_v=initial_flip_v,
        )

        def on_dialog_response(dialog, response_id):
            global _session_config

            if response_id == "start":
                config = dialog.get_config()
                _session_config = config
                logger.debug(
                    "Saved session config: spacing=%.3f, merge_lines=%s, "
                    "rotations=%d, flip_h=%s, flip_v=%s",
                    config.spacing,
                    config.merge_lines,
                    config.rotations,
                    config.flip_h,
                    config.flip_v,
                )
                execute_nesting(editor, items_to_layout, config)
            dialog.close()

        dialog.connect("response", on_dialog_response)
        dialog.present()

    action.connect("activate", on_activate)
    action_registry.register(
        action_name="layout-nesting",
        action=action,
        addon_name=ADDON_NAME,
        label=_("Auto Layout (Nesting)"),
        icon_name="auto-layout-symbolic",
        shortcut="<Ctrl><Alt>n",
        menu=MenuPlacement(menu_id="arrange", priority=50),
        toolbar=ToolbarPlacement(group="arrange", priority=50),
    )
