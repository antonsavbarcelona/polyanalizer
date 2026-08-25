"""Chainlink RTDS connector unit tests, using the exact payload shape
confirmed live for topic "crypto_prices_twap_sixty".
"""
import json

from poly_analyzer.chainlink_feed import ChainlinkFeed
from poly_analyzer.config import MarketConfig
from poly_analyzer.state import MarketState


def make_feed():
    state = MarketState()
    feed = ChainlinkFeed(MarketConfig(), state)
    return feed, state


def twap_msg(value=100_000.5, ts=1_700_000_000_000, symbol="btc/usd"):
    return json.dumps({
        "connection_id": "abc", "topic": "crypto_prices_twap_sixty", "type": "update",
        "timestamp": ts + 50,
        "payload": {"symbol": symbol, "value": value, "full_accuracy_value": "x", "timestamp": ts, "window_s": 60},
    })


def test_parses_twap_value_and_observation_timestamp():
    """U-CL-01"""
    feed, state = make_feed()
    feed._handle(twap_msg(value=100_123.45, ts=1_700_000_000_000))
    assert state.chainlink_twap == 100_123.45
    assert state.chainlink_twap_observation_ts == 1_700_000_000_000


def test_uses_source_observation_timestamp_not_receive_time():
    """U-CL-02: payload.timestamp (Chainlink observation) must be used,
    not the outer publish timestamp or our own receive time."""
    feed, state = make_feed()
    msg = json.loads(twap_msg(ts=1_700_000_000_000))
    msg["timestamp"] = 1_700_000_099_999  # recv/publish time, much later
    feed._handle(json.dumps(msg))
    assert state.chainlink_twap_observation_ts == 1_700_000_000_000


def test_reference_price_latched_once_at_market_start():
    """U-CL-04"""
    feed, state = make_feed()
    feed._handle(twap_msg(value=100_000.0, ts=1000))
    feed._handle(twap_msg(value=105_000.0, ts=2000))
    assert state.chainlink_reference_price == 100_000.0  # unchanged by later updates
    assert state.chainlink_twap == 105_000.0  # but the live TWAP does move


def test_out_of_order_observation_is_ignored():
    feed, state = make_feed()
    feed._handle(twap_msg(value=105_000.0, ts=2000))
    feed._handle(twap_msg(value=99_000.0, ts=1000))  # stale, arrives late
    assert state.chainlink_twap == 105_000.0


def test_historical_dump_message_is_ignored_not_misparsed():
    """The unfiltered-subscription snapshot shape ({"payload":{"data":[...]}})
    must not be mistaken for a live update."""
    feed, state = make_feed()
    dump = json.dumps({"payload": {"data": [{"timestamp": 1000, "value": 99_000.0}]}})
    feed._handle(dump)
    assert state.chainlink_twap is None


def test_malformed_payload_does_not_crash():
    feed, state = make_feed()
    feed._handle(json.dumps({"payload": {"value": "not-a-number", "timestamp": 1000}}))
    assert state.chainlink_twap is None
    feed._handle("")  # empty message, as seen right after subscribe
