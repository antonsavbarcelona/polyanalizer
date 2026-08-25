"""Executable-VWAP MFE/MAE and time-to-level (IMPLEMENTATION CONTRACT #15-16).

Deliberately NOT best-bid MFE/MAE: every r(t) sample requires a full
size_shares fill on the bid side at that book state, exactly like
compute_signal_response's future_sell_vwap -- a spike with insufficient
depth must not look reachable."""
from __future__ import annotations

import sqlite3
from typing import Any

from poly_analyzer.discovery import extract_levels

from research.discovery_types import DiscoveryEntryMark, LEVELS, SignalPathStats
from research.execution.vwap import full_vwap_fill


def compute_path_stats(conn: sqlite3.Connection, market_id: str, direction: str, entry_mark: DiscoveryEntryMark,
                        stats_horizon_ms: int, market_row: dict[str, Any]) -> SignalPathStats:
    assert entry_mark.status == "EXECUTED" and entry_mark.entry_actual_ts is not None and entry_mark.entry_vwap is not None

    prefix = "up" if direction == "UP" else "down"
    entry_ts = entry_mark.entry_actual_ts
    entry_vwap = entry_mark.entry_vwap
    market_end_ts = market_row.get("end_ts")
    window_end = entry_ts + stats_horizon_ms
    if market_end_ts is not None:
        window_end = min(window_end, market_end_ts)

    rows = conn.execute(
        "SELECT * FROM polymarket_book WHERE market_id=? AND ts>=? AND ts<=? ORDER BY ts ASC",
        (market_id, entry_ts, window_end),
    )

    best_ret, best_ts = None, None
    worst_ret, worst_ts = None, None
    time_to_plus: dict[float, int | None] = {lvl: None for lvl in LEVELS}
    time_to_minus: dict[float, int | None] = {lvl: None for lvl in LEVELS}

    for row in rows.fetchall():
        bid, ask = row[f"{prefix}_best_bid"], row[f"{prefix}_best_ask"]
        if bid is not None and ask is not None and bid > ask:
            continue  # not VALID -- crossed book, skip

        bids = extract_levels(row, prefix, "bid")
        fill = full_vwap_fill(bids, entry_mark.size_shares)
        if fill is None:
            continue  # not executable at this state

        sell_vwap, _filled = fill
        ret = sell_vwap - entry_vwap
        ts = row["ts"]

        if best_ret is None or ret > best_ret:
            best_ret, best_ts = ret, ts
        if worst_ret is None or ret < worst_ret:
            worst_ret, worst_ts = ret, ts
        for lvl in LEVELS:
            if time_to_plus[lvl] is None and ret >= lvl:
                time_to_plus[lvl] = ts - entry_ts
            if time_to_minus[lvl] is None and ret <= -lvl:
                time_to_minus[lvl] = ts - entry_ts

    return SignalPathStats(
        entry_mark_id=entry_mark.entry_mark_id,
        stats_horizon_ms=stats_horizon_ms,
        mfe=best_ret,
        mae=worst_ret,
        time_to_mfe_ms=(best_ts - entry_ts) if best_ts is not None else None,
        time_to_mae_ms=(worst_ts - entry_ts) if worst_ts is not None else None,
        time_to_plus_ms=time_to_plus,
        time_to_minus_ms=time_to_minus,
    )
