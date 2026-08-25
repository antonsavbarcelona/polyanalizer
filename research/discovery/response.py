"""Deterministic fixed-horizon response computation (IMPLEMENTATION
CONTRACT #9-14). Only ever called for EXECUTED entry marks -- an entry that
never happened has no price to markout from, so no SignalResponse row is
produced for it at all (this pipeline's denominator rules, contract #14,
already exclude non-executable entries from every P(R>0) calculation)."""
from __future__ import annotations

import sqlite3
from typing import Any

from poly_analyzer.discovery import extract_levels

from research.discovery.book_lookup import first_valid_state_at_or_after
from research.discovery_types import DiscoveryEntryMark, SignalResponse, SignalSnapshot, deterministic_id
from research.execution.vwap import full_vwap_fill

HORIZONS_MS: tuple[int, ...] = (
    100, 250, 500, 750, 1_000, 1_500, 2_000, 3_000, 5_000, 7_500, 10_000, 15_000, 20_000, 30_000, 60_000,
)


def compute_signal_response(conn: sqlite3.Connection, signal: SignalSnapshot, entry_mark: DiscoveryEntryMark,
                             horizon_ms: int, market_row: dict[str, Any]) -> SignalResponse:
    assert entry_mark.status == "EXECUTED" and entry_mark.entry_actual_ts is not None and entry_mark.entry_vwap is not None

    prefix = "up" if signal.direction == "UP" else "down"
    response_target_ts = entry_mark.entry_actual_ts + horizon_ms
    response_id = deterministic_id(
        "signal_response", {"entry_mark_id": entry_mark.entry_mark_id, "horizon_ms": horizon_ms},
    )
    base = dict(
        response_id=response_id, entry_mark_id=entry_mark.entry_mark_id,
        signal_id=signal.signal_id, signal_config_id=signal.signal_config_id,
        asset=signal.asset, market_id=signal.market_id, direction=signal.direction,
        latency_ms=entry_mark.latency_ms, size_shares=entry_mark.size_shares, horizon_ms=horizon_ms,
        response_target_ts=response_target_ts,
    )

    market_end_ts = market_row.get("end_ts")
    if market_end_ts is not None and response_target_ts > market_end_ts:
        return SignalResponse(
            **base, response_actual_ts=None, response_delay_ms=None,
            future_best_bid=None, future_best_ask=None, future_sell_vwap=None,
            available_bid_liquidity=None, raw_response=None, fee_adjusted_response=None,
            response_positive=None, fee_adjusted_positive=None, status="AFTER_MARKET_END",
        )

    row = first_valid_state_at_or_after(conn, signal.market_id, signal.direction, response_target_ts)
    if row is None:
        return SignalResponse(
            **base, response_actual_ts=None, response_delay_ms=None,
            future_best_bid=None, future_best_ask=None, future_sell_vwap=None,
            available_bid_liquidity=None, raw_response=None, fee_adjusted_response=None,
            response_positive=None, fee_adjusted_positive=None, status="NO_DATA",
        )

    response_actual_ts = row["ts"]
    bids = extract_levels(row, prefix, "bid")
    future_best_bid = row.get(f"{prefix}_best_bid")
    future_best_ask = row.get(f"{prefix}_best_ask")
    available_bid_liquidity = sum(size for _, size in bids)

    fill = full_vwap_fill(bids, entry_mark.size_shares)
    if fill is None:
        return SignalResponse(
            **base, response_actual_ts=response_actual_ts, response_delay_ms=response_actual_ts - response_target_ts,
            future_best_bid=future_best_bid, future_best_ask=future_best_ask, future_sell_vwap=None,
            available_bid_liquidity=available_bid_liquidity, raw_response=None, fee_adjusted_response=None,
            response_positive=None, fee_adjusted_positive=None, status="NOT_SELL_EXECUTABLE",
        )

    future_sell_vwap, _filled = fill
    raw_response = future_sell_vwap - entry_mark.entry_vwap
    fee_adjusted_response = raw_response - entry_mark.entry_fee_per_share

    return SignalResponse(
        **base, response_actual_ts=response_actual_ts, response_delay_ms=response_actual_ts - response_target_ts,
        future_best_bid=future_best_bid, future_best_ask=future_best_ask, future_sell_vwap=future_sell_vwap,
        available_bid_liquidity=available_bid_liquidity, raw_response=raw_response,
        fee_adjusted_response=fee_adjusted_response,
        response_positive=raw_response > 0, fee_adjusted_positive=fee_adjusted_response > 0,
        status="AVAILABLE",
    )
