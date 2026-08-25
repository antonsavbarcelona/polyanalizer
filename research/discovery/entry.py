"""Deterministic entry mark computation (IMPLEMENTATION CONTRACT #6-8)."""
from __future__ import annotations

import sqlite3
from typing import Any

from poly_analyzer.discovery import extract_levels

from research.discovery.book_lookup import first_valid_state_at_or_after
from research.discovery_types import DiscoveryEntryMark, SignalSnapshot, deterministic_id
from research.execution.vwap import full_vwap_fill
from research.fees import FeeModel


def compute_entry_mark(conn: sqlite3.Connection, signal: SignalSnapshot, latency_ms: int, size_shares: float,
                        market_row: dict[str, Any], fee_model: FeeModel) -> DiscoveryEntryMark:
    prefix = "up" if signal.direction == "UP" else "down"
    entry_target_ts = signal.signal_ts + latency_ms
    entry_mark_id = deterministic_id(
        "entry_mark", {"signal_id": signal.signal_id, "latency_ms": latency_ms, "size_shares": size_shares},
    )

    row = first_valid_state_at_or_after(conn, signal.market_id, signal.direction, entry_target_ts)
    if row is None:
        return DiscoveryEntryMark(
            entry_mark_id=entry_mark_id, signal_id=signal.signal_id, latency_ms=latency_ms,
            size_shares=size_shares, entry_target_ts=entry_target_ts, entry_actual_ts=None,
            entry_delay_after_target_ms=None, entry_best_bid=None, entry_best_ask=None,
            entry_vwap=None, entry_slippage=None, available_ask_liquidity=None,
            entry_fee_total=None, entry_fee_per_share=None, status="NO_DATA",
        )

    entry_actual_ts = row["ts"]
    asks = extract_levels(row, prefix, "ask")
    best_bid = row.get(f"{prefix}_best_bid")
    best_ask = row.get(f"{prefix}_best_ask")
    available_ask_liquidity = sum(size for _, size in asks)

    fill = full_vwap_fill(asks, size_shares)
    if fill is None:
        return DiscoveryEntryMark(
            entry_mark_id=entry_mark_id, signal_id=signal.signal_id, latency_ms=latency_ms,
            size_shares=size_shares, entry_target_ts=entry_target_ts, entry_actual_ts=entry_actual_ts,
            entry_delay_after_target_ms=entry_actual_ts - entry_target_ts,
            entry_best_bid=best_bid, entry_best_ask=best_ask,
            entry_vwap=None, entry_slippage=None, available_ask_liquidity=available_ask_liquidity,
            entry_fee_total=None, entry_fee_per_share=None, status="NOT_EXECUTABLE",
        )

    entry_vwap, _filled = fill
    entry_fee_per_share = fee_model.taker_fee_per_share(entry_vwap, market_row)
    entry_slippage = (entry_vwap - best_ask) if best_ask is not None else None

    return DiscoveryEntryMark(
        entry_mark_id=entry_mark_id, signal_id=signal.signal_id, latency_ms=latency_ms,
        size_shares=size_shares, entry_target_ts=entry_target_ts, entry_actual_ts=entry_actual_ts,
        entry_delay_after_target_ms=entry_actual_ts - entry_target_ts,
        entry_best_bid=best_bid, entry_best_ask=best_ask,
        entry_vwap=entry_vwap, entry_slippage=entry_slippage,
        available_ask_liquidity=available_ask_liquidity,
        entry_fee_total=entry_fee_per_share * size_shares, entry_fee_per_share=entry_fee_per_share,
        status="EXECUTED",
    )
