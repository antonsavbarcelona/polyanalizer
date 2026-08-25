"""Regression test for a real bug caught in a live E2E run: the rollover
loop was re-deriving "current market" from the Gamma API on every 5s tick,
and a single transient/empty response for the exact-current-window slug
made find_current_market() fall through to the NEXT window (which always
exists and is never "ended"), causing the app to roll over and then roll
back a few ticks later once the current slug resolved again.
"""
import asyncio
import time

from poly_analyzer.config import CONFIG
from poly_analyzer.main import App
from poly_analyzer.market_discovery import MarketInfo


class FakeRecorder:
    def __init__(self):
        self.rows = []

    def enqueue(self, table, row):
        self.rows.append((table, row))


class FakePoly:
    def __init__(self):
        self.tokens = None

    def set_tokens(self, up, down):
        self.tokens = (up, down)


def make_market_info(market_id: str, ends_in_s: float) -> MarketInfo:
    now_ms = int(time.time() * 1000)
    return MarketInfo(
        market_id=market_id, condition_id=f"c-{market_id}", slug=f"slug-{market_id}",
        up_token_id=f"UP-{market_id}", down_token_id=f"DOWN-{market_id}",
        start_ts_ms=now_ms - 60_000, end_ts_ms=now_ms + int(ends_in_s * 1000),
        tick_size=0.01,
    )


def make_app() -> App:
    app = App(CONFIG)
    app.recorder = FakeRecorder()
    app.poly = FakePoly()
    return app


def test_first_call_adopts_whatever_market_is_discovered():
    app = make_app()
    m1 = make_market_info("M1", ends_in_s=400.0)

    def discover(cfg):
        return m1

    asyncio.run(app._maybe_rollover(discover=discover))
    assert app.current_market_id == "M1"
    assert app.state.market_id == "M1"


def test_does_not_re_query_while_current_market_still_has_time_left():
    app = make_app()
    app.current_market_id = "M1"
    app.state.market_id = "M1"
    app.state.market_end_ts = int(time.time() * 1000) + 400_000  # plenty of time left

    def discover_should_not_be_called(cfg):
        raise AssertionError("find_current_market must not be called while current market is still live")

    asyncio.run(app._maybe_rollover(discover=discover_should_not_be_called))
    assert app.current_market_id == "M1"  # unchanged


def test_a_flaky_lookup_of_the_current_slug_does_not_cause_premature_rollover():
    """The actual bug: discover() spuriously returning the NEXT window
    while the current one is still live must not happen anymore, because
    we now only ask when our own end_ts says the current market is done."""
    app = make_app()
    app.current_market_id = "M1"
    app.state.market_id = "M1"
    app.state.market_end_ts = int(time.time() * 1000) + 400_000

    calls = []

    def discover(cfg):
        calls.append(1)
        return make_market_info("M2_NEXT_WINDOW", ends_in_s=1300.0)

    asyncio.run(app._maybe_rollover(discover=discover))
    assert calls == []  # never even asked
    assert app.current_market_id == "M1"


def test_rolls_over_once_the_known_end_ts_has_actually_passed():
    app = make_app()
    app.current_market_id = "M1"
    app.state.market_id = "M1"
    app.state.market_end_ts = int(time.time() * 1000) - 1_000  # already ended
    m2 = make_market_info("M2", ends_in_s=900.0)

    def discover(cfg):
        return m2

    asyncio.run(app._maybe_rollover(discover=discover))
    assert app.current_market_id == "M2"
    assert app.poly.tokens == ("UP-M2", "DOWN-M2")
    assert any(t == "markets" and r["market_id"] == "M2" for t, r in app.recorder.rows)
