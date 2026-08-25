"""Metric aggregation for research TradeResult rows."""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Callable, Iterable

from research.types import StrategyMetrics, TradeResult


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def compute_strategy_metrics(
    strategy_id: str,
    trades: Iterable[TradeResult],
    *,
    signal_count: int | None = None,
    entry_count: int | None = None,
    not_executable_count: int | None = None,
    asset: str | None = None,
    market_id: str | None = None,
) -> StrategyMetrics:
    rows = list(trades)
    trade_count = len(rows)
    if signal_count is None:
        signal_count = len({row.signal_id for row in rows})
    if entry_count is None:
        entry_count = trade_count
    if not_executable_count is None:
        not_executable_count = max(signal_count - entry_count, 0)

    gross = [row.gross_pnl_per_share for row in rows]
    net = [row.net_pnl_per_share for row in rows]
    net_sorted = sorted(net)
    wins = [value for value in net if value > 0]
    losses = [value for value in net if value < 0]
    breakevens = [value for value in net if value == 0]
    entry_slippage = [row.entry_slippage for row in rows if row.entry_slippage is not None]
    exit_slippage = [row.exit_slippage for row in rows if row.exit_slippage is not None]
    pnl_by_market: dict[str, float] = defaultdict(float)
    for row in rows:
        pnl_by_market[row.market_id] += row.pnl_total
    total_abs_pnl = sum(abs(value) for value in pnl_by_market.values())
    top_abs_pnl = max((abs(value) for value in pnl_by_market.values()), default=0.0)

    avg_win = _mean(wins)
    avg_loss = _mean(losses)
    loss_sum = abs(sum(losses))
    win_sum = sum(wins)
    if loss_sum > 0:
        profit_factor = win_sum / loss_sum
    elif win_sum > 0:
        profit_factor = math.inf
    else:
        profit_factor = None

    return StrategyMetrics(
        strategy_id=strategy_id,
        asset=asset,
        market_id=market_id,
        signal_count=signal_count,
        entry_count=entry_count,
        not_executable_count=not_executable_count,
        trade_count=trade_count,
        win_count=len(wins),
        loss_count=len(losses),
        win_rate=(len(wins) / trade_count) if trade_count else None,
        tp_hit_rate=(sum(1 for row in rows if row.tp_hit) / trade_count) if trade_count else None,
        sl_hit_rate=(sum(1 for row in rows if row.sl_hit) / trade_count) if trade_count else None,
        timeout_rate=(sum(1 for row in rows if row.exit_reason == "TIMEOUT") / trade_count) if trade_count else None,
        mean_gross_pnl=_mean(gross),
        median_gross_pnl=median(gross) if gross else None,
        mean_net_pnl=_mean(net),
        median_net_pnl=median(net) if net else None,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=(avg_win / abs(avg_loss)) if avg_win is not None and avg_loss not in (None, 0) else None,
        profit_factor=profit_factor,
        p25=_percentile(net_sorted, 0.25),
        p50=_percentile(net_sorted, 0.50),
        p75=_percentile(net_sorted, 0.75),
        mfe_mean=_mean([row.mfe for row in rows if row.mfe is not None]),
        mae_mean=_mean([row.mae for row in rows if row.mae is not None]),
        holding_time_mean=_mean([float(row.holding_ms) for row in rows]),
        holding_time_median=median([row.holding_ms for row in rows]) if rows else None,
        breakeven_count=len(breakevens),
        mean_entry_slippage=_mean(entry_slippage),
        median_entry_slippage=median(entry_slippage) if entry_slippage else None,
        mean_exit_slippage=_mean(exit_slippage),
        median_exit_slippage=median(exit_slippage) if exit_slippage else None,
        pnl_concentration_top_market=(top_abs_pnl / total_abs_pnl) if total_abs_pnl else None,
    )


def metrics_by_asset(strategy_id: str, trades: Iterable[TradeResult]) -> list[StrategyMetrics]:
    return _grouped_metrics(strategy_id, trades, lambda row: row.asset or "UNKNOWN", asset_group=True)


def metrics_by_market(strategy_id: str, trades: Iterable[TradeResult]) -> list[StrategyMetrics]:
    return _grouped_metrics(strategy_id, trades, lambda row: row.market_id, asset_group=False)


def _grouped_metrics(
    strategy_id: str,
    trades: Iterable[TradeResult],
    key_fn: Callable[[TradeResult], str],
    *,
    asset_group: bool,
) -> list[StrategyMetrics]:
    groups: dict[str, list[TradeResult]] = defaultdict(list)
    for row in trades:
        groups[key_fn(row)].append(row)

    out = []
    for key in sorted(groups):
        out.append(
            compute_strategy_metrics(
                strategy_id,
                groups[key],
                asset=key if asset_group else None,
                market_id=None if asset_group else key,
            )
        )
    return out
