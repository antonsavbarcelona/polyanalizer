"""Single-pass response + path-stats computation (task: "single-pass response/
path processing" perf blocker -- must ship before any wide run).

response.py's compute_signal_response() and path_stats.py's
compute_path_stats() each issue their OWN SQL query and re-walk the future
book path independently -- called once per (signal, latency, size) PER
horizon, that's up to 15+15=30 redundant queries/walks over the same
future path for one entry. This module fetches that path exactly ONCE and
derives every fixed-horizon response AND every stats-horizon's MFE/MAE/
time-to-level from that single materialized list.

Semantics are REQUIRED to stay byte-for-byte identical to the two
independent functions (see tests/test_signal_discovery_path_walk_regression.py)
-- this is a performance rewrite, not a behavior change:
  - "VALID" (not crossed) filtering is identical (book_lookup's rule).
  - Fixed-horizon response still means "the first VALID row at/after
    entry_actual_ts + horizon" (regardless of whether IT alone can fill
    the full size -- NOT_SELL_EXECUTABLE if not, never search further).
  - MFE/MAE/time-to-level still only count rows that are BOTH valid AND
    fully fillable at the requested size (skip everything else silently),
    windowed to [entry_actual_ts, min(entry_actual_ts+stats_horizon,
    market_end_ts)] inclusive.

The key correctness fact that makes single-pass valid: running max/min
over an executable-point sequence is monotonic in time, and "first time a
level is crossed" is a single earliest timestamp -- so every horizon's
answer can be read off one full walk by binary-searching where that
horizon's window ends, rather than re-walking from scratch.
"""
from __future__ import annotations

import bisect
import sqlite3
from typing import Any

from poly_analyzer.discovery import extract_levels

from research.discovery_types import (
    DiscoveryEntryMark,
    LEVELS,
    SignalPathStats,
    SignalResponse,
    SignalSnapshot,
    deterministic_id,
)
from research.execution.vwap import full_vwap_fill


def compute_signal_path(
    conn: sqlite3.Connection, signal: SignalSnapshot, entry_mark: DiscoveryEntryMark,
    market_row: dict[str, Any], response_horizons: tuple[int, ...], stats_horizons: tuple[int, ...],
) -> tuple[list[SignalResponse], list[SignalPathStats]]:
    assert entry_mark.status == "EXECUTED" and entry_mark.entry_actual_ts is not None and entry_mark.entry_vwap is not None

    prefix = "up" if signal.direction == "UP" else "down"
    entry_ts = entry_mark.entry_actual_ts
    entry_vwap = entry_mark.entry_vwap
    market_end_ts = market_row.get("end_ts")
    size_shares = entry_mark.size_shares

    # ONE fetch, unbounded above -- matches first_valid_state_at_or_after's
    # own unbounded query shape exactly, just issued once instead of once
    # per horizon (fixed-horizon responses are allowed to reach past any
    # single stats_horizon's window if that's where the first valid row is).
    raw_rows = conn.execute(
        "SELECT * FROM polymarket_book WHERE market_id=? AND ts>=? ORDER BY ts ASC",
        (signal.market_id, entry_ts),
    ).fetchall()
    rows = [dict(r) for r in raw_rows]

    valid_rows: list[dict] = []
    for row in rows:
        bid, ask = row.get(f"{prefix}_best_bid"), row.get(f"{prefix}_best_ask")
        if bid is not None and ask is not None and bid > ask:
            continue  # crossed -- not VALID
        valid_rows.append(row)
    valid_ts = [r["ts"] for r in valid_rows]

    # ---- fixed-horizon responses (contract #9-14) ----
    responses: list[SignalResponse] = []
    for horizon_ms in response_horizons:
        response_target_ts = entry_ts + horizon_ms
        response_id = deterministic_id(
            "signal_response", {"entry_mark_id": entry_mark.entry_mark_id, "horizon_ms": horizon_ms},
        )
        base = dict(
            response_id=response_id, entry_mark_id=entry_mark.entry_mark_id,
            signal_id=signal.signal_id, signal_config_id=signal.signal_config_id,
            asset=signal.asset, market_id=signal.market_id, direction=signal.direction,
            latency_ms=entry_mark.latency_ms, size_shares=size_shares, horizon_ms=horizon_ms,
            response_target_ts=response_target_ts,
        )

        if market_end_ts is not None and response_target_ts > market_end_ts:
            responses.append(SignalResponse(
                **base, response_actual_ts=None, response_delay_ms=None,
                future_best_bid=None, future_best_ask=None, future_sell_vwap=None,
                available_bid_liquidity=None, raw_response=None, fee_adjusted_response=None,
                response_positive=None, fee_adjusted_positive=None, status="AFTER_MARKET_END",
            ))
            continue

        idx = bisect.bisect_left(valid_ts, response_target_ts)
        if idx >= len(valid_rows):
            responses.append(SignalResponse(
                **base, response_actual_ts=None, response_delay_ms=None,
                future_best_bid=None, future_best_ask=None, future_sell_vwap=None,
                available_bid_liquidity=None, raw_response=None, fee_adjusted_response=None,
                response_positive=None, fee_adjusted_positive=None, status="NO_DATA",
            ))
            continue

        row = valid_rows[idx]
        response_actual_ts = row["ts"]
        bids = extract_levels(row, prefix, "bid")
        future_best_bid = row.get(f"{prefix}_best_bid")
        future_best_ask = row.get(f"{prefix}_best_ask")
        available_bid_liquidity = sum(size for _, size in bids)

        fill = full_vwap_fill(bids, size_shares)
        if fill is None:
            responses.append(SignalResponse(
                **base, response_actual_ts=response_actual_ts,
                response_delay_ms=response_actual_ts - response_target_ts,
                future_best_bid=future_best_bid, future_best_ask=future_best_ask, future_sell_vwap=None,
                available_bid_liquidity=available_bid_liquidity, raw_response=None, fee_adjusted_response=None,
                response_positive=None, fee_adjusted_positive=None, status="NOT_SELL_EXECUTABLE",
            ))
            continue

        future_sell_vwap, _filled = fill
        raw_response = future_sell_vwap - entry_vwap
        fee_adjusted_response = raw_response - entry_mark.entry_fee_per_share
        responses.append(SignalResponse(
            **base, response_actual_ts=response_actual_ts,
            response_delay_ms=response_actual_ts - response_target_ts,
            future_best_bid=future_best_bid, future_best_ask=future_best_ask,
            future_sell_vwap=future_sell_vwap,
            available_bid_liquidity=available_bid_liquidity, raw_response=raw_response,
            fee_adjusted_response=fee_adjusted_response,
            response_positive=raw_response > 0, fee_adjusted_positive=fee_adjusted_response > 0,
            status="AVAILABLE",
        ))

    # ---- MFE/MAE/time-to-level (contract #15-16): ONE forward walk ----
    max_stats_horizon = max(stats_horizons)
    window_end_max = entry_ts + max_stats_horizon
    if market_end_ts is not None:
        window_end_max = min(window_end_max, market_end_ts)

    exec_ts: list[int] = []
    running_best: list[float] = []
    running_best_ts: list[int] = []
    running_worst: list[float] = []
    running_worst_ts: list[int] = []
    plus_first_ts: dict[float, int | None] = {lvl: None for lvl in LEVELS}
    minus_first_ts: dict[float, int | None] = {lvl: None for lvl in LEVELS}

    best = worst = None
    best_ts = worst_ts = None
    for row in valid_rows:
        ts = row["ts"]
        if ts > window_end_max:
            break  # valid_rows is ts-ascending; nothing further matters for any stats horizon
        bids = extract_levels(row, prefix, "bid")
        fill = full_vwap_fill(bids, size_shares)
        if fill is None:
            continue
        sell_vwap, _filled = fill
        ret = sell_vwap - entry_vwap

        if best is None or ret > best:
            best, best_ts = ret, ts
        if worst is None or ret < worst:
            worst, worst_ts = ret, ts
        for lvl in LEVELS:
            if plus_first_ts[lvl] is None and ret >= lvl:
                plus_first_ts[lvl] = ts
            if minus_first_ts[lvl] is None and ret <= -lvl:
                minus_first_ts[lvl] = ts

        exec_ts.append(ts)
        running_best.append(best)
        running_best_ts.append(best_ts)
        running_worst.append(worst)
        running_worst_ts.append(worst_ts)

    path_stats_list: list[SignalPathStats] = []
    for stats_horizon_ms in stats_horizons:
        window_end = entry_ts + stats_horizon_ms
        if market_end_ts is not None:
            window_end = min(window_end, market_end_ts)

        idx = bisect.bisect_right(exec_ts, window_end) - 1
        if idx < 0:
            mfe = mae = None
            time_to_mfe_ms = time_to_mae_ms = None
        else:
            mfe, mae = running_best[idx], running_worst[idx]
            time_to_mfe_ms = running_best_ts[idx] - entry_ts
            time_to_mae_ms = running_worst_ts[idx] - entry_ts

        time_to_plus_ms = {}
        time_to_minus_ms = {}
        for lvl in LEVELS:
            t_plus = plus_first_ts[lvl]
            time_to_plus_ms[lvl] = (t_plus - entry_ts) if (t_plus is not None and t_plus <= window_end) else None
            t_minus = minus_first_ts[lvl]
            time_to_minus_ms[lvl] = (t_minus - entry_ts) if (t_minus is not None and t_minus <= window_end) else None

        path_stats_list.append(SignalPathStats(
            entry_mark_id=entry_mark.entry_mark_id, stats_horizon_ms=stats_horizon_ms,
            mfe=mfe, mae=mae, time_to_mfe_ms=time_to_mfe_ms, time_to_mae_ms=time_to_mae_ms,
            time_to_plus_ms=time_to_plus_ms, time_to_minus_ms=time_to_minus_ms,
        ))

    return responses, path_stats_list
