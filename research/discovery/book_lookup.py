"""Shared Polymarket book-state lookup for the discovery pipeline: strictly
"first VALID state at or after target_ts" (contract #6, #9) -- never
nearest, never last-before. VALID means the relevant token's best_bid does
not exceed its best_ask whenever both are present (matches the crossed-book
skip already used by research/execution/episode.py's TP/SL entry lookup);
a crossed row is skipped, not treated as a data gap."""
from __future__ import annotations

import sqlite3


def first_valid_state_at_or_after(conn: sqlite3.Connection, market_id: str, direction: str,
                                   target_ts: int) -> dict | None:
    prefix = "up" if direction == "UP" else "down"
    rows = conn.execute(
        "SELECT * FROM polymarket_book WHERE market_id=? AND ts>=? ORDER BY ts ASC",
        (market_id, target_ts),
    ).fetchall()
    for row in rows:
        d = dict(row)
        bid, ask = d.get(f"{prefix}_best_bid"), d.get(f"{prefix}_best_ask")
        if bid is not None and ask is not None and bid > ask:
            continue
        return d
    return None
