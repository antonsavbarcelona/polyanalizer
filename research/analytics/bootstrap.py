"""Market-level bootstrap confidence intervals for strategy metrics."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from research.analytics.metrics import _percentile
from research.types import BootstrapResult, TradeResult


def _group_by_market(trades: Iterable[TradeResult]) -> dict[str, list[TradeResult]]:
    groups: dict[str, list[TradeResult]] = defaultdict(list)
    for row in trades:
        groups[row.market_id].append(row)
    return dict(groups)


def _metric_value(trades: list[TradeResult], metric: str) -> float | None:
    if not trades:
        return None
    if metric == "mean_net_pnl":
        return sum(row.net_pnl_per_share for row in trades) / len(trades)
    if metric == "win_rate":
        return sum(1 for row in trades if row.net_pnl_per_share > 0) / len(trades)
    raise ValueError(f"unsupported bootstrap metric: {metric!r}")


def bootstrap_metric_samples(
    trades: Iterable[TradeResult],
    metric: str,
    *,
    iterations: int = 1_000,
    seed: int | None = 0,
) -> list[float]:
    groups = _group_by_market(trades)
    market_ids = sorted(groups)
    if not market_ids or iterations <= 0:
        return []

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_rows: list[TradeResult] = []
        for _ in market_ids:
            market_id = rng.choice(market_ids)
            sampled_rows.extend(groups[market_id])
        value = _metric_value(sampled_rows, metric)
        if value is not None:
            samples.append(value)
    return samples


def bootstrap_confidence_intervals(
    trades: Iterable[TradeResult],
    strategy_id: str,
    *,
    metrics: tuple[str, ...] = ("mean_net_pnl", "win_rate"),
    iterations: int = 1_000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> list[BootstrapResult]:
    rows = list(trades)
    alpha = (1.0 - confidence) / 2.0
    out: list[BootstrapResult] = []
    for metric in metrics:
        samples = sorted(bootstrap_metric_samples(rows, metric, iterations=iterations, seed=seed))
        out.append(
            BootstrapResult(
                strategy_id=strategy_id,
                metric=metric,
                confidence=confidence,
                low=_percentile(samples, alpha),
                high=_percentile(samples, 1.0 - alpha),
                iterations=iterations,
            )
        )
    return out
