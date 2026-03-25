"""
Dialog for configuring nesting layout settings.
"""

import gettext
import logging
from pathlib import Path

from gi.repository import Adw, Gtk

from rayforge.ui_gtk.icons import get_icon
from rayforge.ui_gtk.shared.patched_dialog_window import PatchedMessageDialog
from rayforge.ui_gtk.shared.unit_spin_row import UnitSpinRowHelper
from .deepnest.models import NestConfig

_localedir = Path(__file__).parent.parent / "locales"
_t = gettext.translation("deepnest", localedir=_localedir, fallback=True)
_ = _t.gettext

logger = logging.getLogger(__name__)


class NestingSettingsDialog(PatchedMessageDialog):
    def __init__(
        self,
        parent,
        initial_spacing: float = 0.1,
        initial_constrain_rotation: bool = False,
        initial_flip_h: bool = False,
        initial_flip_v: bool = False,
        **kwargs,
    ):
        super().__init__(
            transient_for=parent,
            modal=True,
            heading=_("Nesting Layout Settings"),
            **kwargs,
        )
        self.add_response("cancel", _("Cancel"))
        self.add_response("start", _("Start Nesting"))
        self.set_default_response("start")
        self.set_close_response("cancel")

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        warning_box = Gtk.Box(
            hexpand=True,
            halign=Gtk.Align.FILL,
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
        )
        warning_box.add_css_class("warning-box")
        warning_box.set_margin_start(6)
        warning_box.set_margin_end(6)
        warning_icon = get_icon("warning-symbolic")
        warning_icon.add_css_class("warning")
        warning_label = Gtk.Label(
            label=_(
                "Nesting can consume significant memory and take a long time "
                "depending on the number and complexity of shapes. It is "
                "recommended to start with a small number of shapes before "
                "scaling up."
            ),
            wrap=True,
            hexpand=True,
            halign=Gtk.Align.FILL,
        )
        warning_label.add_css_class("warning-label")
        warning_box.append(warning_icon)
        warning_box.append(warning_label)
        main_box.append(warning_box)

        group = Adw.PreferencesGroup()

        spacing_adj = Gtk.Adjustment(
            lower=0.0,
            upper=50.0,
            step_increment=0.1,
            page_increment=1.0,
        )
        self.spacing_row = Adw.SpinRow(
            title=_("Spacing"),
            subtitle=_("Distance between nested shapes"),
            adjustment=spacing_adj,
            digits=2,
        )
        group.add(self.spacing_row)

        self.spacing_helper = UnitSpinRowHelper(
            spin_row=self.spacing_row,
            quantity="length",
            max_value_in_base=50.0,
        )
        self.spacing_helper.set_value_in_base_units(initial_spacing)

        self.constrain_rotation_row = Adw.SwitchRow(
            title=_("Constrain Rotation"),
            subtitle=_("Keep parts in their original orientation"),
        )
        self.constrain_rotation_row.set_active(initial_constrain_rotation)
        group.add(self.constrain_rotation_row)

        self.flip_h_row = Adw.SwitchRow(
            title=_("Allow Horizontal Flip"),
            subtitle=_("Mirror parts horizontally for better fit"),
        )
        self.flip_h_row.set_active(initial_flip_h)
        group.add(self.flip_h_row)

        self.flip_v_row = Adw.SwitchRow(
            title=_("Allow Vertical Flip"),
            subtitle=_("Mirror parts vertically for better fit"),
        )
        self.flip_v_row.set_active(initial_flip_v)
        group.add(self.flip_v_row)

        main_box.append(group)

        self.set_extra_child(main_box)

    def get_spacing(self) -> float:
        value = self.spacing_helper.get_value_in_base_units()
        logger.debug("get_spacing: helper returned %.3f", value)
        return value

    def get_constrain_rotation(self) -> bool:
        value = self.constrain_rotation_row.get_active()
        logger.debug("get_constrain_rotation: row returned %s", value)
        return value

    def get_flip_h(self) -> bool:
        value = self.flip_h_row.get_active()
        logger.debug("get_flip_h: row returned %s", value)
        return value

    def get_flip_v(self) -> bool:
        value = self.flip_v_row.get_active()
        logger.debug("get_flip_v: row returned %s", value)
        return value

    def get_config(self) -> NestConfig:
        spacing = self.get_spacing()
        constrain_rotation = self.get_constrain_rotation()
        flip_h = self.get_flip_h()
        flip_v = self.get_flip_v()
        rotations = 1 if constrain_rotation else 36
        logger.debug(
            "Dialog returning config: spacing=%.3f, rotations=%d, "
            "flip_h=%s, flip_v=%s",
            spacing,
            rotations,
            flip_h,
            flip_v,
        )
        return NestConfig(
            spacing=spacing,
            rotations=rotations,
            flip_h=flip_h,
            flip_v=flip_v,
        )
