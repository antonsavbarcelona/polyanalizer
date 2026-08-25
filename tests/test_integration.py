"""Integration tests wiring real connector parsing + state + features +
signal engine together (not just the pure signal_engine unit tests), plus
event-ordering / no-lookahead guarantees.
"""
import json

from poly_analyzer.binance_feed import BinanceFeed
from poly_analyzer.config import CONFIG, MarketConfig
from poly_analyzer.features import compute_features
from poly_analyzer.polymarket_feed import PolymarketFeed
from poly_analyzer.signal_engine import SignalEngine
from poly_analyzer.state import MarketState
from tests.test_polymarket_feed import best_bid_ask_event, book_event

NOW = 1_700_000_000_000


def test_i01_binance_events_flow_through_to_features():
    """I-01: Binance trades/books -> state -> momentum/flow/imbalance chain."""
    state = MarketState()
    feed = BinanceFeed(MarketConfig(), state)

    t = NOW
    feed.handle_message(json.dumps({
        "stream": "btcusdt@bookTicker", "data": {"u": 1, "s": "BTCUSDT", "b": "100000.0", "B": "1", "a": "100002.0", "A": "1"},
    }))
    feed.handle_message(json.dumps({
        "stream": "btcusdt@aggTrade",
        "data": {"a": 1, "p": "100001.0", "q": "0.5", "T": t, "m": False},
    }))
    feed.handle_message(json.dumps({
        "stream": "btcusdt@depth5@100ms",
        "data": {"lastUpdateId": 1, "bids": [["100000.0", "5"]], "asks": [["100002.0", "2"]]},
    }))

    assert state.btc_mid == 100001.0
    assert state.trade_buffer[-1][1] == 0.5
    features = compute_features(state, now_ms=t + 10, cfg=CONFIG.signal)
    assert features.book_imbalance is not None and features.book_imbalance > 0
    assert features.flow_1s == 1.0  # single all-buy trade


def test_i02_poly_book_events_flow_through_with_correct_mapping():
    """I-02"""
    state = MarketState()
    feed = PolymarketFeed(MarketConfig(), state)
    feed.set_tokens("UP", "DOWN")
    feed._handle_raw(json.dumps([
        book_event("UP", [(0.48, 10)], [(0.50, 10)]),
        book_event("DOWN", [(0.49, 10)], [(0.51, 10)]),
    ]))
    assert state.up.bid == 0.48 and state.up.ask == 0.50
    assert state.down.bid == 0.49 and state.down.ask == 0.51
    assert abs(state.up.spread - 0.02) < 1e-9


def _wired_up_state(poly_repriced: bool):
    """Builds a state that satisfies every UP confirmation via the real
    state-update entry points, with explicit control over event timing."""
    state = MarketState()
    state.market_id = "M1"
    state.up_token_id, state.down_token_id = "UP", "DOWN"
    state.market_end_ts = NOW + 400_000
    state.tick_size = 0.01

    for i in range(40):
        # alternating sign so variance (and thus sigma_1s) is nonzero
        state.second_returns.append(0.00005 if i % 2 == 0 else -0.00003)

    # Binance: price rises sharply over the last 250ms/1s.
    state.price_history.append(NOW - 1000, 100_000.0)
    state.price_history.append(NOW - 250, 100_000.0)
    state.price_history.append(NOW, 100_120.0)
    state.trade_buffer.append((NOW, 10.0, True))  # all-buy -> flow=+1.0
    state.binance_bids = [(100_119.0, 60.0)]
    state.binance_asks = [(100_121.0, 20.0)]  # imbalance = (60-20)/80 = +0.5

    poly = PolymarketFeed(MarketConfig(), state)
    poly.set_tokens("UP", "DOWN")
    # ask 500ms ago:
    old_ask = 0.46 if poly_repriced else 0.50
    poly._handle_raw(json.dumps([best_bid_ask_event("UP", 0.49, old_ask, ts=NOW - 600)]))
    # current ask:
    poly._handle_raw(json.dumps([best_bid_ask_event("UP", 0.49, 0.50, ts=NOW)]))
    return state


def test_i03_binance_impulse_with_flat_poly_produces_up_signal():
    """I-03: real move + confirmations, Polymarket hasn't repriced -> exactly one UP signal."""
    state = _wired_up_state(poly_repriced=False)
    features = compute_features(state, now_ms=NOW, cfg=CONFIG.signal)
    engine = SignalEngine(CONFIG.signal)
    signal = engine.evaluate(state, features, NOW)
    assert signal is not None
    assert signal.direction == "UP"


def test_i04_binance_impulse_with_poly_already_repriced_produces_no_signal():
    """I-04: same Binance move, but Polymarket UP ask already jumped +4c -> no edge, no signal."""
    state = _wired_up_state(poly_repriced=True)
    features = compute_features(state, now_ms=NOW, cfg=CONFIG.signal)
    engine = SignalEngine(CONFIG.signal)
    signal = engine.evaluate(state, features, NOW)
    assert signal is None


def test_i10_market_rollover_does_not_leak_old_state():
    state = MarketState()
    state.market_id = "OLD"
    state.up_token_id, state.down_token_id = "OLD_UP", "OLD_DOWN"
    state.up.bid, state.up.ask = 0.40, 0.41
    state.up.ask_history.append(NOW, 0.41)
    state.chainlink_reference_price = 99_000.0

    state.reset_for_new_market("NEW", "cond2", "slug2", "NEW_UP", "NEW_DOWN",
                                NOW, NOW + 900_000, 0.01)

    assert state.market_id == "NEW"
    assert state.up_token_id == "NEW_UP"
    assert state.up.bid is None and state.up.ask is None
    assert len(state.up.ask_history) == 0
    assert state.chainlink_reference_price is None
    # Binance-side state (not market-specific) is intentionally untouched.


def test_i_time_01_out_of_order_binance_event_does_not_rewrite_the_past():
    """I-TIME-01 / no-lookahead: a delayed event with an old exchange_ts,
    arriving late, must not retroactively change already-computed state."""
    state = MarketState()
    state.on_binance_depth(NOW, [(100_000.0, 1.0)], [(100_002.0, 1.0)])

    # A stale event with an old timestamp arrives after we've already moved on.
    state.on_binance_trade(NOW - 5_000, 99_000.0, 1.0, is_buyer_maker=False)
    state.price_history.append(NOW - 5_000, 99_000.0)  # would corrupt ordering if not guarded

    # The time series must not re-order around the late-arriving stale point.
    latest = state.price_history.latest
    assert latest[0] == NOW  # the stale sample did not become "latest"
