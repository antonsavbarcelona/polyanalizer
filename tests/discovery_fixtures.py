"""Shared helpers for building small synthetic raw-data SQLite DBs, driven
through the REAL live feed classes (handle_message/_handle_event are pure
sync functions, no socket needed) so raw rows are byte-identical in shape
to what the live collector actually writes.
"""
from __future__ import annotations

import json

from poly_analyzer.binance_feed import BinanceFeed
from poly_analyzer.chainlink_feed import ChainlinkFeed
from poly_analyzer.config import MarketConfig, RecordingConfig
from poly_analyzer.db import Recorder, WriteJob
from poly_analyzer.polymarket_feed import PolymarketFeed
from poly_analyzer.state import MarketState


class CollectingRecorder:
    """Records enqueue() calls in-memory; not connected to a real queue."""

    def __init__(self):
        self.jobs: list[WriteJob] = []

    def enqueue(self, table, row):
        self.jobs.append(WriteJob(table, dict(row)))


def persist(tmp_path, jobs: list[WriteJob]):
    from dataclasses import replace as dc_replace
    from poly_analyzer.config import RecorderConfig

    cfg = dc_replace(RecorderConfig(), db_path=str(tmp_path / "test.db"))
    rec = Recorder(cfg)
    rec.connect()
    rec._write_batch(jobs)
    return rec


def agg_trade_msg(agg_id, ts, price, qty, is_buyer_maker):
    return json.dumps({"stream": "btcusdt@aggTrade",
                        "data": {"a": agg_id, "p": str(price), "q": str(qty), "T": ts, "m": is_buyer_maker}})


def depth_msg(bids, asks):
    return json.dumps({"stream": "btcusdt@depth10@100ms",
                        "data": {"lastUpdateId": 1,
                                  "bids": [[str(p), str(s)] for p, s in bids],
                                  "asks": [[str(p), str(s)] for p, s in asks]}})


def build_binance_only_db(tmp_path, market_id: str, events: list[tuple]):
    """events: list of ("trade", agg_id, ts, price, qty, is_buyer_maker) or
    ("depth", bids, asks) tuples, fed in order through a real BinanceFeed
    into both a live MarketState and a persisted raw DB."""
    state = MarketState()
    state.market_id = market_id
    collector = CollectingRecorder()
    feed = BinanceFeed(MarketConfig(), state, recorder=collector)

    for evt in events:
        if evt[0] == "trade":
            _, agg_id, ts, price, qty, is_buyer_maker = evt
            feed.handle_message(agg_trade_msg(agg_id, ts, price, qty, is_buyer_maker))
        elif evt[0] == "depth":
            _, bids, asks = evt
            feed.handle_message(depth_msg(bids, asks))

    rec = persist(tmp_path, collector.jobs)
    return state, rec


def make_recorder(tmp_path):
    from dataclasses import replace as dc_replace
    from poly_analyzer.config import RecorderConfig

    cfg = dc_replace(RecorderConfig(), db_path=str(tmp_path / "test.db"))
    rec = Recorder(cfg)
    rec.connect()
    return rec


def insert_rows(rec: Recorder, table: str, rows: list[dict]) -> None:
    rec._write_batch([WriteJob(table, r) for r in rows])


def binance_book_row(market_id, ts, bids, asks, levels=10) -> dict:
    """bids/asks: list of (price, size), best-first. Directly matches the
    binance_book schema -- used where tests need full control over
    timestamps (BinanceFeed.handle_message always stamps book rows with
    real wall-clock time, which precise timing tests can't control)."""
    row = {
        "market_id": market_id, "exchange_ts": None, "recv_ts": ts, "recv_monotonic_ns": ts * 1_000_000,
        "best_bid_price": bids[0][0] if bids else None, "best_bid_size": bids[0][1] if bids else None,
        "best_ask_price": asks[0][0] if asks else None, "best_ask_size": asks[0][1] if asks else None,
        "is_debug": 0,
    }
    for i in range(levels):
        bp, bs = bids[i] if i < len(bids) else (None, None)
        ap, asz = asks[i] if i < len(asks) else (None, None)
        row[f"bid_{i + 1}_price"], row[f"bid_{i + 1}_size"] = bp, bs
        row[f"ask_{i + 1}_price"], row[f"ask_{i + 1}_size"] = ap, asz
    return row


def binance_trade_row(market_id, agg_id, exchange_ts, price, qty, aggressor_side) -> dict:
    return {"market_id": market_id, "agg_trade_id": agg_id, "exchange_ts": exchange_ts,
            "recv_ts": exchange_ts, "recv_monotonic_ns": exchange_ts * 1_000_000,
            "price": price, "qty": qty, "aggressor_side": aggressor_side, "is_debug": 0}


def poly_book_row(market_id, ts, up_bid=None, up_ask=None, up_bids=None, up_asks=None,
                   down_bid=None, down_ask=None, down_bids=None, down_asks=None, levels=10) -> dict:
    up_bids, up_asks = up_bids or [], up_asks or []
    down_bids, down_asks = down_bids or [], down_asks or []
    row = {
        "market_id": market_id, "ts": ts, "recv_monotonic_ns": ts * 1_000_000, "is_debug": 0,
        "up_best_bid": up_bid, "up_best_bid_size": up_bids[0][1] if up_bids else None,
        "up_best_ask": up_ask, "up_best_ask_size": up_asks[0][1] if up_asks else None,
        "down_best_bid": down_bid, "down_best_bid_size": down_bids[0][1] if down_bids else None,
        "down_best_ask": down_ask, "down_best_ask_size": down_asks[0][1] if down_asks else None,
    }
    for prefix, bids, asks in (("up", up_bids, up_asks), ("down", down_bids, down_asks)):
        for i in range(levels):
            bp, bs = bids[i] if i < len(bids) else (None, None)
            ap, asz = asks[i] if i < len(asks) else (None, None)
            row[f"{prefix}_bid_{i + 1}_price"], row[f"{prefix}_bid_{i + 1}_size"] = bp, bs
            row[f"{prefix}_ask_{i + 1}_price"], row[f"{prefix}_ask_{i + 1}_size"] = ap, asz
    return row


def market_row(market_id, up_token_id="UP", down_token_id="DOWN", start_ts=0, end_ts=10**9,
               tick_size=0.01) -> dict:
    return {"market_id": market_id, "up_token_id": up_token_id, "down_token_id": down_token_id,
            "start_ts": start_ts, "end_ts": end_ts, "tick_size": tick_size}


def build_poly_db(tmp_path, market_id: str, up_id: str, down_id: str, book_events: list[dict]):
    """book_events: list of dicts {"direction": "UP"/"DOWN", "ts": int,
    "bids": [...], "asks": [...]} fed as CLOB "book" events."""
    state = MarketState()
    state.market_id = market_id
    collector = CollectingRecorder()
    feed = PolymarketFeed(MarketConfig(), state, recording=RecordingConfig(poly_heartbeat_ms=100_000),
                           recorder=collector)
    feed.set_tokens(up_id, down_id)

    for evt in book_events:
        token = up_id if evt["direction"] == "UP" else down_id
        raw = {
            "event_type": "book", "asset_id": token, "timestamp": str(evt["ts"]),
            "bids": [{"price": str(p), "size": str(s)} for p, s in evt["bids"]],
            "asks": [{"price": str(p), "size": str(s)} for p, s in evt["asks"]],
        }
        feed._handle_event(raw, recv_monotonic_ns=evt["ts"] * 1_000_000)

    rec = persist(tmp_path, collector.jobs)
    return state, rec, feed
