from poly_analyzer.config import SignalConfig
from poly_analyzer.features import book_imbalance_top5, sigma_1s, trade_flow
from poly_analyzer.state import MarketState
from poly_analyzer.timeseries import TimeSeries, time_return


def test_momentum_is_time_based_not_tick_based():
    """U-FEAT-05: irregular tick spacing must not be read as 'N ticks ago'."""
    series = TimeSeries(max_age_ms=10_000)
    # Ticks arrive at wildly irregular spacing.
    series.append(0, 100.0)
    series.append(50, 100.5)
    series.append(900, 100.8)
    series.append(1_000, 101.0)  # exactly 1000ms after t=0

    # return_1s(now=1000) must compare against the sample AT ts=0 (the only
    # one <= 1000-1000=0), not "the price 4 ticks ago" (which would be 100.5).
    r = time_return(series, now_ts_ms=1_000, lookback_ms=1_000)
    assert r == (101.0 - 100.0) / 100.0


def test_momentum_not_ready_returns_none_not_zero():
    """U-FEAT-04: insufficient history -> None, never a silent 0."""
    series = TimeSeries(max_age_ms=10_000)
    series.append(0, 100.0)
    series.append(100, 100.1)
    assert time_return(series, now_ts_ms=100, lookback_ms=1_000) is None


def test_momentum_uses_nearest_before_not_future_sample_across_a_gap():
    """U-FEAT-09 analogue: with a sparse/gappy series, "1s ago" must resolve
    to the nearest sample at-or-before the cutoff, never a later one -- even
    when that means reaching further back than 1s because nothing closer
    exists."""
    series = TimeSeries(max_age_ms=200_000)
    series.append(0, 100.0)
    series.append(150_000, 130.0)  # huge gap: nothing between t=0 and t=150000
    series.append(150_500, 130.2)
    # cutoff for a 1s lookback at now=150500 is 149500; the only sample
    # at-or-before that is the one at t=0, NOT the closer-but-later t=150000.
    r = time_return(series, now_ts_ms=150_500, lookback_ms=1_000)
    assert abs(r - (130.2 - 100.0) / 100.0) < 1e-9


def test_trade_flow_predominantly_buys():
    state = MarketState()
    state.trade_buffer.extend([(0, 75.0, True), (0, 25.0, False)])
    assert trade_flow(state, now_ms=1000, window_ms=1000) == 0.5


def test_trade_flow_no_trades_returns_none_not_divide_by_zero():
    state = MarketState()
    assert trade_flow(state, now_ms=1000, window_ms=1000) is None


def test_trade_flow_excludes_trades_outside_window():
    state = MarketState()
    state.trade_buffer.append((0, 100.0, True))  # outside window
    state.trade_buffer.append((5_900, 10.0, False))
    result = trade_flow(state, now_ms=6_000, window_ms=1_000)
    assert result == -1.0  # only the sell trade counted


def test_book_imbalance_top5():
    state = MarketState()
    state.binance_bids = [(100.0, 300.0), (99.0, 300.0)]
    state.binance_asks = [(101.0, 200.0), (102.0, 200.0)]
    assert abs(book_imbalance_top5(state) - 0.2) < 1e-9


def test_book_imbalance_empty_side_returns_none():
    state = MarketState()
    state.binance_bids = [(100.0, 10.0)]
    state.binance_asks = []
    assert book_imbalance_top5(state) is None


def test_sigma_1s_requires_minimum_samples():
    cfg = SignalConfig()
    state = MarketState()
    for i in range(cfg.sigma_min_samples - 1):
        state.second_returns.append(0.0001 * i)
    assert sigma_1s(state, cfg) is None


def test_sigma_1s_constant_price_is_zero():
    cfg = SignalConfig()
    state = MarketState()
    for _ in range(cfg.sigma_min_samples):
        state.second_returns.append(0.0)
    assert sigma_1s(state, cfg) == 0.0
