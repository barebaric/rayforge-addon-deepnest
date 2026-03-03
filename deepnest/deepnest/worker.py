import logging
from typing import List, Dict, Any, Optional, Callable

from .models import NestConfig, SheetInfo
from .placement import NestResult, place_parts

logger = logging.getLogger(__name__)


def batch_nest_worker(
    individuals: List[Dict[str, Any]],
    parts: List[Dict[str, Any]],
    sheets: List[SheetInfo],
    config_dict: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Optional[NestResult]]:
    config = NestConfig(**config_dict)
    results = []

    for i, individual in enumerate(individuals):
        rotations = individual["rotation"]
        result = place_parts(parts, sheets, rotations, config)
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, len(individuals))

    return results
