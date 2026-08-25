from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import MarketConfig


@dataclass
class MarketInfo:
    market_id: str
    condition_id: str
    slug: str
    up_token_id: str
    down_token_id: str
    start_ts_ms: int
    end_ts_ms: int
    tick_size: float
    resolution_source: str | None = None
    maker_base_fee: float | None = None
    taker_base_fee: float | None = None
    fee_rate: float | None = None
    fee_exponent: float | None = None


def _window_start(now_s: float, window_s: int) -> int:
    return int(now_s // window_s) * window_s


def _fetch_market_by_slug(cfg: MarketConfig, slug: str) -> dict | None:
    url = f"{cfg.gamma_api_base}/markets?slug={slug}"
    req = urllib.request.Request(url, headers={"User-Agent": "poly_analyzer/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    if not data:
        return None
    return data[0]


def _parse_market(raw: dict, window_s: int) -> MarketInfo:
    end_dt = datetime.fromisoformat(raw["endDate"].replace("Z", "+00:00")).astimezone(timezone.utc)
    end_ts_ms = int(end_dt.timestamp() * 1000)
    start_ts_ms = end_ts_ms - window_s * 1000
    clob_ids = json.loads(raw["clobTokenIds"])
    tick = float(raw.get("orderPriceMinTickSize") or 0.01)
    fee_schedule = raw.get("feeSchedule") or {}
    return MarketInfo(
        market_id=str(raw["id"]),
        condition_id=raw["conditionId"],
        slug=raw["slug"],
        up_token_id=clob_ids[0],
        down_token_id=clob_ids[1],
        start_ts_ms=start_ts_ms,
        end_ts_ms=end_ts_ms,
        tick_size=tick,
        resolution_source=raw.get("resolutionSource"),
        maker_base_fee=_as_float(raw.get("makerBaseFee")),
        taker_base_fee=_as_float(raw.get("takerBaseFee")),
        fee_rate=_as_float(fee_schedule.get("rate")),
        fee_exponent=_as_float(fee_schedule.get("exponent")),
    )


def _as_float(v) -> float | None:
    return None if v is None else float(v)


def find_current_market(cfg: MarketConfig) -> MarketInfo | None:
    """Locates the active 15m BTC Up/Down window via the Gamma API.

    Slugs are deterministic: f"{prefix}-{window_start_unix}" where
    window_start is the 900s-aligned epoch boundary (confirmed live against
    gamma-api.polymarket.com). Tries current window, then next (in case the
    listing lags right at a rollover), then previous as a last resort.
    """
    now_s = time.time()
    base = _window_start(now_s, cfg.window_s)
    for offset in (0, cfg.window_s, -cfg.window_s):
        slug = f"{cfg.asset_slug_prefix}-{base + offset}"
        raw = _fetch_market_by_slug(cfg, slug)
        if raw is None:
            continue
        info = _parse_market(raw, cfg.window_s)
        if info.end_ts_ms > now_s * 1000:
            return info
    return None
