"""Polymarket CLOB market-channel WebSocket connector.

Wire format confirmed live against wss://ws-subscriptions-clob.polymarket.com/ws/market:
subscribe {"assets_ids":[...], "type":"market", "custom_feature_enabled":true},
heartbeat PING every 10s (server replies literal "PONG"), events carry
event_type in {"book","price_change","best_bid_ask","last_trade_price",
"tick_size_change","new_market","market_resolved"}.

Raw persistence: a polymarket_book row (top-10 levels, BOTH UP and DOWN) is
written whenever top-of-book actually changes on either side, plus a
heartbeat at cfg.poly_heartbeat_ms so a quiet market still leaves a trail.
Event-driven, not fixed-interval-only, specifically so a short spike that
reverts between heartbeats (e.g. bid .53 -> .58 -> .53 inside 250ms) is not
lost -- confirmed live that writing only on a fixed interval misses this.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets

from .config import MarketConfig, RecordingConfig
from .state import MarketState

log = logging.getLogger(__name__)
PING_INTERVAL_S = 10
BOOK_LEVELS = 10


class PolymarketFeed:
    def __init__(self, cfg: MarketConfig, state: MarketState, recording: RecordingConfig | None = None,
                 recorder=None, on_update=None, is_debug: bool = False):
        self.cfg = cfg
        self.recording = recording or RecordingConfig()
        self.state = state
        self.recorder = recorder
        self.on_update = on_update
        self.is_debug = is_debug
        self._tokens: tuple[str, str] | None = None
        self._book_snapshot: dict[str, tuple[list[tuple[float, float]], list[tuple[float, float]]]] = {
            "UP": ([], []),
            "DOWN": ([], []),
        }
        self._resubscribe = asyncio.Event()
        self.connected = False

        self._last_written_top_of_book: tuple | None = None
        self._last_raw_write_ms = 0

    def set_tokens(self, up_id: str, down_id: str) -> None:
        self._tokens = (up_id, down_id)
        self._book_snapshot = {"UP": ([], []), "DOWN": ([], [])}
        self._last_written_top_of_book = None
        self._resubscribe.set()

    def book_levels(self, direction: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Returns (bids desc, asks asc) from the last full book snapshot."""
        return self._book_snapshot.get(direction, ([], []))

    async def run(self) -> None:
        while True:
            if self._tokens is None:
                await asyncio.sleep(0.2)
                continue
            try:
                await self._connect_and_stream()
            except Exception:
                log.exception("polymarket feed error, reconnecting")
                self.connected = False
                await asyncio.sleep(2)

    async def _connect_and_stream(self) -> None:
        async with websockets.connect(self.cfg.clob_ws_url, ping_interval=None) as ws:
            await self._subscribe(ws)
            self._resubscribe.clear()
            self.connected = True
            ping_task = asyncio.create_task(self._ping_loop(ws))
            try:
                async for raw in ws:
                    if self._resubscribe.is_set():
                        await self._subscribe(ws)
                        self._resubscribe.clear()
                    if raw == "PONG":
                        continue
                    self._handle_raw(raw)
            finally:
                ping_task.cancel()
                self.connected = False

    async def _subscribe(self, ws) -> None:
        up_id, down_id = self._tokens
        await ws.send(json.dumps({
            "assets_ids": [up_id, down_id],
            "type": "market",
            "custom_feature_enabled": True,
        }))

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL_S)
            try:
                await ws.send("PING")
            except Exception:
                return

    def _handle_raw(self, raw: str) -> None:
        recv_monotonic_ns = time.monotonic_ns()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        for evt in (data if isinstance(data, list) else [data]):
            try:
                self._handle_event(evt, recv_monotonic_ns)
            except (KeyError, ValueError, TypeError):
                log.warning("dropping malformed polymarket event: %r", str(evt)[:200])

    def _direction_for(self, asset_id: str | None) -> str | None:
        if not self._tokens or asset_id is None:
            return None
        up_id, down_id = self._tokens
        if asset_id == up_id:
            return "UP"
        if asset_id == down_id:
            return "DOWN"
        return None

    def _handle_event(self, evt: dict, recv_monotonic_ns: int) -> None:
        event_type = evt.get("event_type")
        recv_ts = int(time.time() * 1000)
        direction = self._direction_for(evt.get("asset_id"))

        if event_type == "book" and direction:
            ts = int(evt.get("timestamp", recv_ts))
            bids = sorted(
                ((float(l["price"]), float(l["size"])) for l in evt.get("bids", [])),
                key=lambda x: -x[0],
            )
            asks = sorted(
                ((float(l["price"]), float(l["size"])) for l in evt.get("asks", [])),
                key=lambda x: x[0],
            )
            self._book_snapshot[direction] = (bids, asks)
            book = self.state.token_book(direction)
            self.state.on_poly_book(
                direction, ts,
                bids[0][0] if bids else None, asks[0][0] if asks else None,
                bids[0][1] if bids else None, asks[0][1] if asks else None,
            )
            tick_size = evt.get("tick_size")
            if tick_size is not None:
                self.state.tick_size = float(tick_size)
            last_trade = evt.get("last_trade_price")
            if last_trade is not None:
                self.state.on_poly_trade(direction, float(last_trade), None)

        elif event_type == "last_trade_price" and direction:
            ts = int(evt.get("timestamp", recv_ts))
            price = evt.get("price")
            size = evt.get("size")
            if price is not None:
                self.state.on_poly_trade(direction, float(price), float(size) if size is not None else None)

        elif event_type == "best_bid_ask" and direction:
            ts = int(evt.get("timestamp", recv_ts))
            bid = evt.get("best_bid")
            ask = evt.get("best_ask")
            book = self.state.token_book(direction)
            self.state.on_poly_book(
                direction, ts,
                float(bid) if bid is not None else book.bid,
                float(ask) if ask is not None else book.ask,
                book.bid_size, book.ask_size,
            )

        elif event_type == "price_change":
            # Unlike "book"/"best_bid_ask", a single price_change message has
            # NO top-level asset_id -- each entry in price_changes[] carries
            # its own, and one message can bundle both UP and DOWN changes.
            ts = int(evt.get("timestamp", recv_ts))
            for pc in evt.get("price_changes", []):
                pc_direction = self._direction_for(pc.get("asset_id"))
                if pc_direction is None:
                    continue
                book = self.state.token_book(pc_direction)
                bb, ba = pc.get("best_bid"), pc.get("best_ask")
                bid = float(bb) if bb is not None else book.bid
                ask = float(ba) if ba is not None else book.ask
                self.state.on_poly_book(pc_direction, ts, bid, ask, book.bid_size, book.ask_size)

        self._maybe_record_raw(recv_ts, recv_monotonic_ns)

        if self.on_update:
            self.on_update("polymarket")

    def _maybe_record_raw(self, now: int, recv_monotonic_ns: int) -> None:
        if self.recorder is None or self.state.market_id is None:
            return
        top_of_book = (self.state.up.bid, self.state.up.ask, self.state.down.bid, self.state.down.ask)
        due = now - self._last_raw_write_ms >= self.recording.poly_heartbeat_ms
        changed = top_of_book != self._last_written_top_of_book
        if not changed and not due:
            return
        self._last_raw_write_ms = now
        self._last_written_top_of_book = top_of_book
        self.recorder.enqueue("polymarket_book", build_polymarket_book_row(
            self.state, self, now, recv_monotonic_ns, self.is_debug,
        ))


def build_polymarket_book_row(state: MarketState, feed: "PolymarketFeed", ts: int, recv_monotonic_ns: int,
                               is_debug: bool, levels: int = BOOK_LEVELS) -> dict:
    row = {"market_id": state.market_id, "ts": ts, "recv_monotonic_ns": recv_monotonic_ns,
           "is_debug": 1 if is_debug else 0}
    for direction, prefix in (("UP", "up"), ("DOWN", "down")):
        book = state.token_book(direction)
        bids, asks = feed.book_levels(direction)
        row[f"{prefix}_best_bid"] = book.bid
        row[f"{prefix}_best_bid_size"] = book.bid_size
        row[f"{prefix}_best_ask"] = book.ask
        row[f"{prefix}_best_ask_size"] = book.ask_size
        for i in range(levels):
            bp, bs = bids[i] if i < len(bids) else (None, None)
            ap, asz = asks[i] if i < len(asks) else (None, None)
            row[f"{prefix}_bid_{i + 1}_price"] = bp
            row[f"{prefix}_bid_{i + 1}_size"] = bs
            row[f"{prefix}_ask_{i + 1}_price"] = ap
            row[f"{prefix}_ask_{i + 1}_size"] = asz
    return row
