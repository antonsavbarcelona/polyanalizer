"""Polymarket RTDS: Chainlink TWAP feed (wss://ws-live-data.polymarket.com).

Confirmed live against the official Polymarket/real-time-data-client wire
protocol: {"action":"subscribe","subscriptions":[{"topic":...,"type":"update",
"filters":"<compact JSON>"}]}. The server matches `filters` as an exact
string, NOT parsed JSON -- json.dumps' default spacing ({"symbol": "x"})
silently never matches, so filters must be built with separators=(",", ":").
Topic is "crypto_prices_twap_sixty" (60s window, used by 15m/4h markets) or
"crypto_prices_twap_thirty" (30s, 5m markets). Client heartbeat is the
literal string "ping" every 5s.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets

from .config import MarketConfig
from .state import MarketState

log = logging.getLogger(__name__)
PING_INTERVAL_S = 5


def _topic_for_window(window_s: int) -> str:
    return "crypto_prices_twap_sixty" if window_s == 60 else "crypto_prices_twap_thirty"


class ChainlinkFeed:
    def __init__(self, cfg: MarketConfig, state: MarketState, recorder=None, on_update=None,
                 is_debug: bool = False):
        self.cfg = cfg
        self.state = state
        self.recorder = recorder
        self.on_update = on_update
        self.is_debug = is_debug
        self.connected = False
        self._topic = _topic_for_window(cfg.chainlink_twap_window_s)
        self._filters = json.dumps({"symbol": cfg.chainlink_symbol}, separators=(",", ":"))

    async def run(self) -> None:
        while True:
            try:
                async with websockets.connect(self.cfg.rtds_ws_url, ping_interval=None) as ws:
                    await self._subscribe(ws)
                    self.connected = True
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for raw in ws:
                            self._handle(raw)
                    finally:
                        ping_task.cancel()
            except Exception:
                log.exception("chainlink feed error, reconnecting")
            self.connected = False
            await asyncio.sleep(2)

    async def _subscribe(self, ws) -> None:
        await ws.send(json.dumps({
            "action": "subscribe",
            "subscriptions": [
                {"topic": self._topic, "type": "update", "filters": self._filters},
            ],
        }))

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL_S)
            try:
                await ws.send("ping")
            except Exception:
                return

    def _handle(self, raw: str) -> None:
        recv_monotonic_ns = time.monotonic_ns()
        if not raw:
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        payload = msg.get("payload")
        if not isinstance(payload, dict) or "value" in payload and "data" in payload:
            return
        if "value" not in payload or "timestamp" not in payload:
            return
        recv_ts = int(time.time() * 1000)
        try:
            observation_ts_ms = int(payload["timestamp"])
            value = float(payload["value"])
        except (TypeError, ValueError):
            log.warning("dropping malformed chainlink payload: %r", str(payload)[:200])
            return
        self.state.on_chainlink_twap(observation_ts_ms, value)
        if self.recorder is not None:
            self.recorder.enqueue("chainlink_observations", {
                "market_id": self.state.market_id,
                "observation_ts": observation_ts_ms,
                "recv_ts": recv_ts,
                "recv_monotonic_ns": recv_monotonic_ns,
                "twap_value": value,
                "twap_window": self.cfg.chainlink_twap_window_s,
                "is_debug": 1 if self.is_debug else 0,
            })
        if self.on_update:
            self.on_update("chainlink")
