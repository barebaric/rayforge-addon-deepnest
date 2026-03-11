"""
Pytest configuration for deepnest builtin addon tests.

This conftest ensures that producers and steps are registered
with their respective registries before tests run.
"""

from typing import cast

import pytest
from rayforge.doceditor.layout.registry import layout_registry


def _register_layout():
    """Register all producers from deepnest addon."""
    from deepnest.nesting import NestingLayoutStrategy

    layout_registry.register(NestingLayoutStrategy, name="deepnest")


@pytest.fixture(scope="session", autouse=True)
def register_deepnest():
    """
    Automatically register deepnest producers and steps
    for all tests in this addon.

    This also prevents the context from loading addons by setting
    _addon_mgr to a sentinel, which prevents the property from
    creating a real AddonManager.
    """
    from rayforge.context import get_context
    from rayforge.addon_mgr.addon_manager import AddonManager

    context = get_context()
    context._addon_mgr = cast(AddonManager, object())

    _register_layout()
    yield
