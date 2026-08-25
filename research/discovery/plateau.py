"""Parameter-neighbor detection + plateau score (IMPLEMENTATION CONTRACT
#36-37). Config A and B are neighbors ONLY if they differ in exactly one
ordinal grid axis, by exactly one adjacent step in that axis's sorted grid
values, with every other axis held exactly equal -- never "close enough" on
multiple axes at once."""
from __future__ import annotations

import math
from dataclasses import dataclass

from research.discovery_types import SignalDiscoveryConfig


def _grid_index(grid: dict[str, list], axis: str, value) -> int | None:
    values = sorted(v for v in grid.get(axis, []) if v is not None)
    try:
        return values.index(value)
    except ValueError:
        return None


def is_neighbor(a: SignalDiscoveryConfig, b: SignalDiscoveryConfig, grid: dict[str, list]) -> bool:
    if a.id == b.id:
        return False
    diffs = [field for field in grid if getattr(a, field, None) != getattr(b, field, None)]
    if len(diffs) != 1:
        return False
    axis = diffs[0]
    idx_a = _grid_index(grid, axis, getattr(a, axis))
    idx_b = _grid_index(grid, axis, getattr(b, axis))
    if idx_a is None or idx_b is None:
        return False
    return abs(idx_a - idx_b) == 1


def find_neighbors(target: SignalDiscoveryConfig, candidates: list[SignalDiscoveryConfig],
                    grid: dict[str, list]) -> list[SignalDiscoveryConfig]:
    return [c for c in candidates if is_neighbor(target, c, grid)]


@dataclass
class PlateauMetrics:
    neighbor_count: int
    neighbor_positive_count: int
    neighbor_positive_ratio: float | None
    central_uplift_mean: float | None
    neighbor_mean_uplift: float | None
    neighbor_std_uplift: float | None
    plateau_score: float | None


def compute_plateau_metrics(central_uplift_mean: float | None, signal_count: int,
                             neighbor_uplifts: list[float | None]) -> PlateauMetrics:
    valid = [u for u in neighbor_uplifts if u is not None]
    neighbor_count = len(neighbor_uplifts)
    neighbor_positive_count = sum(1 for u in valid if u > 0)
    neighbor_positive_ratio = (neighbor_positive_count / neighbor_count) if neighbor_count else None

    neighbor_mean_uplift = (sum(valid) / len(valid)) if valid else None
    if len(valid) >= 2:
        m = neighbor_mean_uplift
        neighbor_std_uplift = (sum((v - m) ** 2 for v in valid) / (len(valid) - 1)) ** 0.5
    else:
        neighbor_std_uplift = None

    score = None
    if central_uplift_mean is not None and neighbor_positive_ratio is not None:
        std_term = neighbor_std_uplift if neighbor_std_uplift is not None else 0.0
        score = (
            central_uplift_mean
            * math.sqrt(math.log(1 + max(signal_count, 0)))
            * neighbor_positive_ratio
            / (1.0 + std_term)
        )

    return PlateauMetrics(
        neighbor_count=neighbor_count, neighbor_positive_count=neighbor_positive_count,
        neighbor_positive_ratio=neighbor_positive_ratio, central_uplift_mean=central_uplift_mean,
        neighbor_mean_uplift=neighbor_mean_uplift, neighbor_std_uplift=neighbor_std_uplift,
        plateau_score=score,
    )
