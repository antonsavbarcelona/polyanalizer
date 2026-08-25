"""Settlement tracking: after a market ends, wait for Polymarket's OFFICIAL
resolution and store it as ground truth, alongside a self-computed
derived_outcome (recorded Chainlink TWAP at market end vs reference) as a
diagnostic cross-check only.

A mismatch between official and derived means our understanding of the
reference/TWAP-window rules is wrong -- that must be investigated, never
silently patched over by trusting the derived value instead.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .config import Config

log = logging.getLogger(__name__)


def fetch_market_raw(cfg: Config, market_id: str) -> dict | None:
    url = f"{cfg.market.gamma_api_base}/markets/{market_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "poly_analyzer/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def parse_official_outcome(raw: dict) -> str | None:
    if raw.get("umaResolutionStatus") != "resolved":
        return None
    outcomes = raw.get("outcomes")
    prices = raw.get("outcomePrices")
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    if isinstance(prices, str):
        prices = json.loads(prices)
    if not outcomes or not prices:
        return None
    for name, price in zip(outcomes, prices):
        if float(price) >= 0.5:
            return str(name).upper()
    return None


def parse_resolution_ts(raw: dict) -> int | None:
    iso = raw.get("umaEndDate") or raw.get("closedTime")
    if not iso:
        return None
    try:
        iso = iso.replace("Z", "+00:00").replace(" ", "T", 1)
        return int(datetime.fromisoformat(iso).astimezone(timezone.utc).timestamp() * 1000)
    except ValueError:
        return None


def compute_derived_outcome(db_path: str, market_id: str, end_ts_ms: int,
                             reference_price: float | None) -> tuple[str | None, float | None]:
    """Reads our OWN recorded chainlink_observations (never live state,
    which has moved on to later markets by the time settlement is checked)
    for the TWAP observation at-or-after market end, matching the market's
    own "TWAP at end of window >= price at start" resolution rule.
    """
    if reference_price is None:
        return None, None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT twap_value FROM chainlink_observations "
            "WHERE market_id = ? AND observation_ts >= ? AND twap_value IS NOT NULL "
            "ORDER BY observation_ts ASC LIMIT 1",
            (market_id, end_ts_ms),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT twap_value FROM chainlink_observations "
                "WHERE market_id = ? AND twap_value IS NOT NULL "
                "ORDER BY observation_ts DESC LIMIT 1",
                (market_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return None, None
        twap_at_end = row[0]
        outcome = "UP" if twap_at_end >= reference_price else "DOWN"
        return outcome, twap_at_end
    finally:
        conn.close()


async def track_settlement(cfg: Config, market_id: str, end_ts_ms: int,
                            reference_price: float | None, recorder) -> None:
    await asyncio.sleep(cfg.market.settlement_start_delay_s)

    official_outcome = None
    resolution_ts = None
    deadline = time.time() + cfg.market.settlement_poll_timeout_s
    while time.time() < deadline:
        raw = await asyncio.to_thread(fetch_market_raw, cfg, market_id)
        if raw is not None:
            official_outcome = parse_official_outcome(raw)
            if official_outcome is not None:
                resolution_ts = parse_resolution_ts(raw)
                break
        await asyncio.sleep(cfg.market.settlement_poll_interval_s)

    derived_outcome, twap_at_end = await asyncio.to_thread(
        compute_derived_outcome, cfg.recorder.db_path, market_id, end_ts_ms, reference_price,
    )

    if official_outcome is None:
        log.warning("settlement: no official resolution for market %s within %.0fs timeout",
                     market_id, cfg.market.settlement_poll_timeout_s)
    if official_outcome is not None and derived_outcome is not None and official_outcome != derived_outcome:
        log.error(
            "SETTLEMENT MISMATCH market=%s official=%s derived=%s twap_at_end=%s reference=%s "
            "-- investigate reference/TWAP-window handling before trusting derived_outcome",
            market_id, official_outcome, derived_outcome, twap_at_end, reference_price,
        )

    recorder.enqueue("market_settlement", {
        "market_id": market_id,
        "official_outcome": official_outcome,
        "resolution_ts": resolution_ts,
        "derived_outcome": derived_outcome,
        "derived_twap_at_end": twap_at_end,
        "reference_price": reference_price,
        "checked_at": int(time.time() * 1000),
    })
