"""Research execution helpers layered on top of the collector VWAP primitive."""
from __future__ import annotations

from poly_analyzer.execution import vwap_fill


def full_vwap_fill(levels: list[tuple[float, float]], target_size: float) -> tuple[float, float] | None:
    """Return a VWAP only when the book can fill the full requested size."""
    fill = vwap_fill(levels, target_size)
    if fill is None:
        return None
    vwap, filled = fill
    if filled < target_size:
        return None
    return vwap, filled
