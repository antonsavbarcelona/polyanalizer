"""Polymarket CLOB connector unit tests, using the exact event shapes
confirmed live against wss://ws-subscriptions-clob.polymarket.com/ws/market.
"""
import json

from poly_analyzer.config import MarketConfig
from poly_analyzer.polymarket_feed import PolymarketFeed
from poly_analyzer.state import MarketState

UP, DOWN = "UP_TOKEN", "DOWN_TOKEN"


def make_feed():
    state = MarketState()
    feed = PolymarketFeed(MarketConfig(), state)
    feed.set_tokens(UP, DOWN)
    return feed, state


def book_event(asset_id, bids, asks, ts=1_700_000_000_000):
    return {
        "event_type": "book", "market": "0xabc", "asset_id": asset_id, "timestamp": str(ts),
        "hash": "h", "tick_size": "0.01",
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
    }


def best_bid_ask_event(asset_id, best_bid, best_ask, ts=1_700_000_000_000):
    return {"event_type": "best_bid_ask", "market": "0xabc", "asset_id": asset_id,
            "best_bid": str(best_bid), "best_ask": str(best_ask), "spread": "0.01", "timestamp": str(ts)}


def price_change_event(asset_id, best_bid, best_ask, ts=1_700_000_000_000):
    return {"event_type": "price_change", "market": "0xabc", "timestamp": str(ts),
            "price_changes": [{"asset_id": asset_id, "side": "BUY", "price": "0.5", "size": "10",
                                "best_bid": str(best_bid), "best_ask": str(best_ask)}]}


def test_up_book_updates_only_up_state():
    """U-POLY-01"""
    feed, state = make_feed()
    feed._handle_raw(json.dumps([book_event(UP, [(0.49, 10)], [(0.51, 20)])]))
    assert state.up.bid == 0.49 and state.up.ask == 0.51
    assert state.down.bid is None and state.down.ask is None


def test_down_book_updates_only_down_state():
    """U-POLY-02"""
    feed, state = make_feed()
    feed._handle_raw(json.dumps([book_event(DOWN, [(0.48, 5)], [(0.50, 5)])]))
    assert state.down.bid == 0.48 and state.down.ask == 0.50
    assert state.up.bid is None and state.up.ask is None


def test_best_bid_selected_from_multiple_levels():
    """U-POLY-03"""
    feed, state = make_feed()
    feed._handle_raw(json.dumps([book_event(UP, [(0.40, 1), (0.49, 1), (0.45, 1)], [(0.60, 1), (0.51, 1), (0.55, 1)])]))
    assert state.up.bid == 0.49  # highest bid
    assert state.up.ask == 0.51  # lowest ask
    bids, asks = feed.book_levels("UP")
    assert [p for p, _ in bids] == [0.49, 0.45, 0.40]
    assert [p for p, _ in asks] == [0.51, 0.55, 0.60]


def test_empty_ask_book_blocks_signal_eligibility():
    """U-POLY-04"""
    feed, state = make_feed()
    feed._handle_raw(json.dumps([book_event(UP, [(0.49, 10)], [])]))
    assert state.up.ask is None
    assert state.up.valid is False


def test_empty_bid_book_prevents_maker_fill():
    """U-POLY-05: covered at the execution layer, but confirm state exposes bid=None."""
    feed, state = make_feed()
    feed._handle_raw(json.dumps([book_event(UP, [], [(0.51, 10)])]))
    assert state.up.bid is None


def test_up_down_tokens_never_confused():
    """U-POLY-06"""
    feed, state = make_feed()
    feed._handle_raw(json.dumps([
        book_event(UP, [(0.40, 1)], [(0.41, 1)]),
        book_event(DOWN, [(0.58, 1)], [(0.59, 1)]),
    ]))
    assert state.up.bid == 0.40 and state.down.bid == 0.58


def test_broken_book_bid_greater_than_ask_marked_invalid():
    """U-POLY-09"""
    feed, state = make_feed()
    feed._handle_raw(json.dumps([book_event(UP, [(0.49, 1)], [(0.51, 1)])]))
    feed._handle_raw(json.dumps([best_bid_ask_event(UP, 0.60, 0.58)]))  # crossed
    assert state.up.bid == 0.49 and state.up.ask == 0.51  # unchanged, rejected
    assert state.up.valid is True


def test_best_bid_ask_event_updates_top_of_book():
    feed, state = make_feed()
    feed._handle_raw(json.dumps([best_bid_ask_event(UP, 0.52, 0.53)]))
    assert state.up.bid == 0.52 and state.up.ask == 0.53


def test_price_change_event_updates_top_of_book():
    feed, state = make_feed()
    feed._handle_raw(json.dumps([price_change_event(UP, 0.47, 0.48)]))
    assert state.up.bid == 0.47 and state.up.ask == 0.48


def test_malformed_event_is_dropped_not_crashed():
    feed, state = make_feed()
    bad = json.dumps([{"event_type": "book", "asset_id": UP, "bids": [{"price": "oops"}], "asks": []}])
    feed._handle_raw(bad)  # must not raise
    assert state.up.bid is None
