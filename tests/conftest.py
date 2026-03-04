"""
Pytest configuration for deepnest builtin addon tests.

This conftest ensures that producers and steps are registered
with their respective registries before tests run.
"""

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

    This also prevents ensure_addons_loaded() from loading via
    AddonManager, which would register classes from a different
    module path (rayforge_addons.*) causing isinstance() checks
    to fail in tests.
    """
    from rayforge import worker_init

    worker_init._worker_addons_loaded = True

    _register_layout()
    yield
