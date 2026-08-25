"""Binance connector unit tests, using the exact combined-stream envelope
shape confirmed live: {"stream": "<name>", "data": {...}}.
"""
import json

from poly_analyzer.binance_feed import BinanceFeed
from poly_analyzer.config import MarketConfig
from poly_analyzer.state import MarketState


def make_feed():
    state = MarketState()
    feed = BinanceFeed(MarketConfig(), state)
    return feed, state


def agg_trade_msg(agg_id=1, ts=1_700_000_000_000, price="100000.50", qty="0.01", is_buyer_maker=False):
    return json.dumps({
        "stream": "btcusdt@aggTrade",
        "data": {"e": "aggTrade", "a": agg_id, "p": price, "q": qty, "T": ts, "m": is_buyer_maker},
    })


def book_ticker_msg(bid="99999.00", ask="100001.00"):
    return json.dumps({
        "stream": "btcusdt@bookTicker",
        "data": {"u": 1, "s": "BTCUSDT", "b": bid, "B": "1.0", "a": ask, "A": "1.0"},
    })


def depth_msg(bids=None, asks=None):
    bids = bids or [["99999.00", "1.0"], ["99998.00", "2.0"]]
    asks = asks or [["100001.00", "1.5"], ["100002.00", "0.5"]]
    return json.dumps({"stream": "btcusdt@depth5@100ms", "data": {"lastUpdateId": 1, "bids": bids, "asks": asks}})


def test_agg_trade_parses_price_qty_ts_and_side():
    """U-BIN-01"""
    feed, state = make_feed()
    feed.handle_message(agg_trade_msg(price="100000.50", qty="0.02", ts=1_700_000_000_123, is_buyer_maker=False))
    assert state.btc_last == 100000.50
    ts, qty, is_buy = state.trade_buffer[-1]
    assert ts == 1_700_000_000_123
    assert qty == 0.02
    assert is_buy is True  # isBuyerMaker=False -> taker is the buyer


def test_agg_trade_buyer_maker_means_aggressive_sell():
    feed, state = make_feed()
    feed.handle_message(agg_trade_msg(is_buyer_maker=True))
    _, _, is_buy = state.trade_buffer[-1]
    assert is_buy is False


def test_book_ticker_updates_best_bid_ask():
    """U-BIN-02"""
    feed, state = make_feed()
    feed.handle_message(book_ticker_msg(bid="99999.00", ask="100001.00"))
    assert state.btc_bid == 99999.00
    assert state.btc_ask == 100001.00


def test_depth_levels_sorted_bids_desc_asks_asc_regardless_of_input_order():
    """U-BIN-03"""
    feed, state = make_feed()
    feed.handle_message(depth_msg(
        bids=[["99998.00", "2.0"], ["99999.00", "1.0"]],  # deliberately out of order
        asks=[["100002.00", "0.5"], ["100001.00", "1.5"]],
    ))
    assert [p for p, _ in state.binance_bids] == [99999.00, 99998.00]
    assert [p for p, _ in state.binance_asks] == [100001.00, 100002.00]


def test_invalid_event_missing_price_is_dropped_not_crashed():
    """U-BIN-04"""
    feed, state = make_feed()
    bad = json.dumps({"stream": "btcusdt@aggTrade", "data": {"a": 1, "T": 1, "m": False}})  # no "p", "q"
    feed.handle_message(bad)  # must not raise
    assert state.btc_last is None


def test_duplicate_agg_trade_id_does_not_double_count_volume():
    """U-BIN-05"""
    feed, state = make_feed()
    feed.handle_message(agg_trade_msg(agg_id=5, qty="1.0"))
    feed.handle_message(agg_trade_msg(agg_id=5, qty="1.0"))  # exact duplicate delivery
    assert len(state.trade_buffer) == 1


def test_mid_price_computation():
    """U-BIN-07"""
    feed, state = make_feed()
    feed.handle_message(book_ticker_msg(bid="100.00", ask="102.00"))
    assert state.btc_mid == 101.00


def test_crossed_book_is_rejected_and_state_stays_valid():
    """U-BIN-08"""
    feed, state = make_feed()
    feed.handle_message(book_ticker_msg(bid="100.00", ask="102.00"))
    feed.handle_message(book_ticker_msg(bid="105.00", ask="103.00"))  # crossed: bid > ask
    assert state.btc_bid == 100.00 and state.btc_ask == 102.00  # unchanged
    assert state.binance_valid is True
