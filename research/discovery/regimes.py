"""Fixed regime boundaries (contract #28). Never recomputed per SignalConfig
-- TTE/price/spread buckets are hardcoded constants; volatility quantiles
are computed once per asset over the discovery dataset and frozen into
experiment metadata (see VolatilityRegimeBoundaries)."""
from __future__ import annotations

from research.discovery_types import (
    PRICE_REGIME_BOUNDS,
    SPREAD_REGIME_BOUNDS,
    TTE_REGIME_BOUNDS_S,
    VolatilityRegimeBoundaries,
)


def _classify(value: float | None, bounds: tuple[tuple[float, float, str], ...]) -> str | None:
    if value is None:
        return None
    for lo, hi, label in bounds:
        if lo <= value < hi:
            return label
    # last bucket is right-inclusive (contract #28: "кроме последнего bucket")
    last_lo, last_hi, last_label = bounds[-1]
    if value >= last_lo:
        return last_label
    return None


def classify_tte(remaining_s: float | None) -> str | None:
    return _classify(remaining_s, TTE_REGIME_BOUNDS_S)


def classify_price(price: float | None) -> str | None:
    return _classify(price, PRICE_REGIME_BOUNDS)


def classify_spread(spread: float | None) -> str | None:
    return _classify(spread, SPREAD_REGIME_BOUNDS)


def compute_volatility_boundaries(asset: str, samples: list[float]) -> VolatilityRegimeBoundaries | None:
    """P25/P75/P95 of `samples` (e.g. every vol_30s observed while scanning
    the discovery dataset for this asset). None if there's no usable data --
    callers must then leave volatility_regime as None rather than guessing."""
    values = sorted(v for v in samples if v is not None)
    if len(values) < 4:
        return None
    return VolatilityRegimeBoundaries(
        asset=asset,
        p25=_percentile(values, 0.25),
        p75=_percentile(values, 0.75),
        p95=_percentile(values, 0.95),
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac
