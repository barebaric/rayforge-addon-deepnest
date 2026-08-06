"""
Deepnest - Nesting layout functionality.

This builtin addon provides the nesting layout strategy for optimal
arrangement of workpieces using the deepnest algorithm.
"""

from .core import DeepNest
from .models import (
    NestConfig,
    NestSolution,
    Placement,
    SheetInfo,
    WorkpieceInfo,
)

__all__ = [
    "DeepNest",
    "NestConfig",
    "NestSolution",
    "Placement",
    "SheetInfo",
    "WorkpieceInfo",
]
