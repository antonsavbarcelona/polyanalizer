"""SignalEpisode: the (signal, execution_config) future book-path, fetched
from the DB exactly once and then reused for every ExitConfig evaluated
against it (task §22/§70 -- never re-replay/re-query per exit config)."""
from __future__ import annotations

import bisect
import sqlite3
from dataclasses import dataclass

from poly_analyzer.discovery import DiscoveredSignal, ExecutionMark, extract_levels

from research.execution.vwap import full_vwap_fill


@dataclass
class SignalEpisode:
    signal: DiscoveredSignal
    execution_config_id: str
    entry_mark: ExecutionMark
    path_rows: list[dict]  # raw polymarket_book rows, ts >= entry_mark.actual_ts, ASC. Fetched ONCE.
    path_ts: list[int]     # path_rows' "ts" column, pre-extracted for bisect.
    market_row: dict       # for fee-formula inputs (fee_rate, fee_exponent).


def compute_research_entry_mark(
    conn: sqlite3.Connection,
    market_id: str,
    direction: str,
    signal_ts: int,
    latency_ms: int,
    size_shares: float,
    market_end_ts: int | None = None,
) -> tuple[ExecutionMark, str]:
    target_ts = signal_ts + latency_ms
    prefix = "up" if direction == "UP" else "down"
    if market_end_ts is None:
        rows = conn.execute(
            "SELECT * FROM polymarket_book WHERE market_id=? AND ts>=? ORDER BY ts ASC",
            (market_id, target_ts),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM polymarket_book WHERE market_id=? AND ts>=? AND ts<=? ORDER BY ts ASC",
            (market_id, target_ts, market_end_ts),
        ).fetchall()

    saw_book = False
    for row in rows:
        saw_book = True
        bids = sorted(extract_levels(row, prefix, "bid"), key=lambda level: level[0], reverse=True)
        asks = sorted(extract_levels(row, prefix, "ask"), key=lambda level: level[0])
        best_bid = bids[0][0] if bids else row[f"{prefix}_best_bid"]
        best_ask = asks[0][0] if asks else row[f"{prefix}_best_ask"]
        if best_bid is not None and best_ask is not None and best_bid > best_ask:
            continue
        fill = full_vwap_fill(asks, size_shares)
        if fill is None:
            return ExecutionMark(latency_ms, target_ts, row["ts"], None, None, best_ask), "NOT_EXECUTABLE"
        return ExecutionMark(latency_ms, target_ts, row["ts"], fill[0], fill[1], best_ask), "EXECUTED"

    status = "NO_ENTRY_DATA" if not saw_book else "NOT_EXECUTABLE"
    return ExecutionMark(latency_ms, target_ts, None, None, None, None), status


def build_episode(conn: sqlite3.Connection, signal: DiscoveredSignal, execution_config_id: str,
                   latency_ms: int, size_shares: float, market_row: dict) -> SignalEpisode | None:
    """Computes the entry mark (via discovery.compute_execution_mark) and,
    if executable, fetches the full future polymarket_book path once. Returns
    None if entry is NOT_EXECUTABLE (entry_mark.entry_vwap is None -- e.g. no
    book state ever arrived at/after the target entry timestamp, or the
    book had no ask liquidity at that state)."""
    entry_mark, status = compute_research_entry_mark(
        conn,
        signal.market_id,
        signal.direction,
        signal.signal_ts,
        latency_ms,
        size_shares,
        market_row.get("end_ts"),
    )
    if status != "EXECUTED":
        return None
    return build_episode_from_entry(conn, signal, execution_config_id, entry_mark, market_row)


def build_episode_from_entry(
    conn: sqlite3.Connection,
    signal: DiscoveredSignal,
    execution_config_id: str,
    entry_mark: ExecutionMark,
    market_row: dict,
) -> SignalEpisode:
    """Build the reusable future path from an already-computed successful entry mark."""

    # Same query shape as poly_analyzer.discovery.analyze_response: all rows
    # at/after entry, ascending, no upper bound (horizon is applied later by
    # the exit-policy walk, not by this fetch, so a single fetch here can
    # serve every ExitConfig regardless of max_holding_ms).
    if market_row.get("end_ts") is None:
        rows = conn.execute(
            "SELECT * FROM polymarket_book WHERE market_id=? AND ts>=? ORDER BY ts ASC",
            (signal.market_id, entry_mark.actual_ts),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM polymarket_book WHERE market_id=? AND ts>=? AND ts<=? ORDER BY ts ASC",
            (signal.market_id, entry_mark.actual_ts, market_row["end_ts"]),
        ).fetchall()
    path_rows = [dict(r) for r in rows]
    path_ts = [r["ts"] for r in path_rows]

    return SignalEpisode(
        signal=signal,
        execution_config_id=execution_config_id,
        entry_mark=entry_mark,
        path_rows=path_rows,
        path_ts=path_ts,
        market_row=market_row,
    )


def first_row_at_or_after(path_ts: list[int], target_ts: int) -> int | None:
    """bisect over an already-in-memory, ascending `path_ts` list (NOT a new
    SQL query -- `path_ts` comes from a SignalEpisode already fetched once).
    Returns the INDEX into path_rows/path_ts of the first row with
    ts >= target_ts, or None if target_ts is past the last available row."""
    idx = bisect.bisect_left(path_ts, target_ts)
    if idx >= len(path_ts):
        return None
    return idx
