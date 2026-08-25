"""Binance combined WS stream: aggTrade + bookTicker + depth10@100ms.

Confirmed live: wss://stream.binance.com:9443/stream?streams=... wraps each
message as {"stream": "<name>", "data": {...}}. aggTrade carries an exchange
timestamp ("T"); depth10 does not (Binance's partial-depth stream has no
per-update server timestamp), so book rows use receive time only.

Raw persistence is the point: every trade and every ~100ms book snapshot
(top-10 both sides) is written close to as-received, so any feature
definition can be recomputed offline in discovery.py without a new live
run. bookTicker still drives the live in-memory state.btc_bid/ask (it
updates faster than the throttled depth stream, which matters for live
momentum/z-score) but is not itself persisted raw -- best bid/ask from the
depth10 snapshot covers the same information for storage purposes.
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


class BinanceFeed:
    def __init__(self, cfg: MarketConfig, state: MarketState, recorder=None, on_update=None,
                 is_debug: bool = False):
        self.cfg = cfg
        self.state = state
        self.recorder = recorder
        self.on_update = on_update
        self.is_debug = is_debug
        self.connected = False
        self._last_agg_trade_id: int | None = None

    async def run(self) -> None:
        url = self.cfg.binance_ws_url.format(symbol=self.cfg.binance_symbol)
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    self.connected = True
                    async for raw in ws:
                        self.handle_message(raw)
            except Exception:
                log.exception("binance feed error, reconnecting")
            self.connected = False
            await asyncio.sleep(2)

    def handle_message(self, raw: str) -> None:
        """Malformed/incomplete messages are logged and skipped, never
        allowed to corrupt state or kill the connection (U-BIN-04)."""
        try:
            self._handle(raw)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            log.warning("dropping malformed binance message: %r", raw[:200])

    def _handle(self, raw: str) -> None:
        recv_monotonic_ns = time.monotonic_ns()
        msg = json.loads(raw)
        stream = msg.get("stream", "")
        data = msg.get("data", {})
        recv_ts = int(time.time() * 1000)

        if stream.endswith("@aggTrade"):
            self._handle_trade(data, recv_ts, recv_monotonic_ns)
        elif stream.endswith("@bookTicker"):
            self.state.on_binance_book_ticker(recv_ts, float(data["b"]), float(data["a"]))
        elif "depth" in stream:
            self._handle_depth(data, recv_ts, recv_monotonic_ns)
        else:
            return

        if self.on_update:
            self.on_update("binance")

    def _handle_trade(self, data: dict, recv_ts: int, recv_monotonic_ns: int) -> None:
        agg_id = int(data["a"])
        if self._last_agg_trade_id is not None and agg_id <= self._last_agg_trade_id:
            return  # duplicate/replayed trade (U-BIN-05)
        self._last_agg_trade_id = agg_id
        exchange_ts = int(data["T"])
        price = float(data["p"])
        qty = float(data["q"])
        is_buyer_maker = bool(data["m"])
        aggressor_side = "SELL" if is_buyer_maker else "BUY"
        self.state.on_binance_trade(exchange_ts, price, qty, is_buyer_maker)
        if self.recorder is not None:
            self.recorder.enqueue("binance_trades", {
                "market_id": self.state.market_id,
                "agg_trade_id": agg_id,
                "exchange_ts": exchange_ts,
                "recv_ts": recv_ts,
                "recv_monotonic_ns": recv_monotonic_ns,
                "price": price,
                "qty": qty,
                "aggressor_side": aggressor_side,
                "is_debug": 1 if self.is_debug else 0,
            })

    def _handle_depth(self, data: dict, recv_ts: int, recv_monotonic_ns: int) -> None:
        bids = sorted(((float(p), float(q)) for p, q in data.get("bids", [])), key=lambda x: -x[0])
        asks = sorted(((float(p), float(q)) for p, q in data.get("asks", [])), key=lambda x: x[0])
        self.state.on_binance_depth(recv_ts, bids, asks)
        if self.recorder is not None:
            self.recorder.enqueue("binance_book", build_binance_book_row(
                bids, asks, self.state.market_id, None, recv_ts, recv_monotonic_ns, self.is_debug,
            ))


def build_binance_book_row(bids: list[tuple[float, float]], asks: list[tuple[float, float]],
                            market_id, exchange_ts, recv_ts: int, recv_monotonic_ns: int,
                            is_debug: bool, levels: int = 10) -> dict:
    """bids/asks must already be sorted best-first (desc/asc respectively)."""
    row = {
        "market_id": market_id,
        "exchange_ts": exchange_ts,
        "recv_ts": recv_ts,
        "recv_monotonic_ns": recv_monotonic_ns,
        "best_bid_price": bids[0][0] if bids else None,
        "best_bid_size": bids[0][1] if bids else None,
        "best_ask_price": asks[0][0] if asks else None,
        "best_ask_size": asks[0][1] if asks else None,
        "is_debug": 1 if is_debug else 0,
    }
    for i in range(levels):
        bp, bs = bids[i] if i < len(bids) else (None, None)
        ap, asz = asks[i] if i < len(asks) else (None, None)
        row[f"bid_{i + 1}_price"] = bp
        row[f"bid_{i + 1}_size"] = bs
        row[f"ask_{i + 1}_price"] = ap
        row[f"ask_{i + 1}_size"] = asz
    return row
