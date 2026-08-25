"""Regression test for a real bug caught in a live E2E run: writing a raw
book row on every raw Polymarket WS message (rather than only when
top-of-book actually changed) produced ~194 rows/sec with many duplicate
timestamps/content -- unsustainable for a real collection run.

The throttle now lives in PolymarketFeed itself (POLY-06: event-driven on
change + heartbeat), since it owns the raw persistence decision.
"""
import dataclasses

from poly_analyzer.config import MarketConfig, RecordingConfig
from poly_analyzer.polymarket_feed import PolymarketFeed
from poly_analyzer.state import MarketState


class FakeRecorder:
    def __init__(self):
        self.rows = []

    def enqueue(self, table, row):
        self.rows.append((table, row))


def make_feed(**recording_overrides) -> PolymarketFeed:
    state = MarketState()
    state.market_id = "M1"
    recording = dataclasses.replace(RecordingConfig(), **recording_overrides)
    feed = PolymarketFeed(MarketConfig(), state, recording=recording, recorder=FakeRecorder())
    feed.set_tokens("UP", "DOWN")
    return feed


def test_no_write_when_unchanged_and_not_due():
    feed = make_feed(poly_heartbeat_ms=250)
    feed._maybe_record_raw(now=1000, recv_monotonic_ns=1)
    assert len(feed.recorder.rows) == 1  # first call always writes (no prior state)

    feed._maybe_record_raw(now=1005, recv_monotonic_ns=2)  # nothing changed, 5ms later
    assert len(feed.recorder.rows) == 1  # must not have written again


def test_writes_immediately_when_top_of_book_changes():
    feed = make_feed(poly_heartbeat_ms=250)
    feed._maybe_record_raw(now=1000, recv_monotonic_ns=1)
    assert len(feed.recorder.rows) == 1

    feed.state.up.bid = 0.55  # top-of-book actually moved
    feed._maybe_record_raw(now=1005, recv_monotonic_ns=2)
    assert len(feed.recorder.rows) == 2


def test_heartbeat_writes_even_without_a_change():
    feed = make_feed(poly_heartbeat_ms=250)
    feed._maybe_record_raw(now=1000, recv_monotonic_ns=1)
    assert len(feed.recorder.rows) == 1

    feed._maybe_record_raw(now=1000 + 250, recv_monotonic_ns=2)
    assert len(feed.recorder.rows) == 2


def test_short_spike_between_heartbeats_is_not_lost():
    """POLY-06: t=1.000 bid .53, t=1.080 bid .58, t=1.170 bid .53 -- all
    three must be captured even with a 250ms heartbeat, because each is an
    actual top-of-book change (event-driven), not a heartbeat sample."""
    feed = make_feed(poly_heartbeat_ms=250)
    feed._maybe_record_raw(now=1000, recv_monotonic_ns=1)
    feed.state.up.bid = 0.58
    feed._maybe_record_raw(now=1080, recv_monotonic_ns=2)
    feed.state.up.bid = 0.53
    feed._maybe_record_raw(now=1170, recv_monotonic_ns=3)
    assert len(feed.recorder.rows) == 3
    bids_seen = [r["up_best_bid"] for _, r in feed.recorder.rows]
    assert bids_seen == [None, 0.58, 0.53]  # the spike (.58) was captured, not skipped
