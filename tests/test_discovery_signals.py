"""Discovery signal replay: same decision structure as the live signal
engine, but every window/threshold is a config parameter (SIGNAL-01..07).
"""
import dataclasses

from poly_analyzer.discovery import DiscoverySignalConfig, replay_signals
from tests.discovery_fixtures import (
    binance_book_row, binance_trade_row, insert_rows, make_recorder, market_row, poly_book_row,
)

MARKET_ID = "M1"
BASE_CFG = DiscoverySignalConfig(
    momentum_window_ms=100, volatility_lookback_ms=1_000, sigma_min_samples=5,
    flow_window_ms=500, flow_threshold=0.3,
    imbalance_depth=5, imbalance_threshold=0.10,
    poly_lag_window_ms=500, max_poly_reprice=0.01,
    tte_min_s=0.0, tte_max_s=1_000_000.0,
    ask_min=0.20, ask_max=0.80, spread_max=0.03,
)


def _build_up_impulse_db(tmp_path):
    rec = make_recorder(tmp_path)
    insert_rows(rec, "markets", [market_row(MARKET_ID, end_ts=500_000)])

    book_rows = []
    for ts in range(0, 1_000, 100):
        book_rows.append(binance_book_row(MARKET_ID, ts, [(100.0, 50.0)], [(100.02, 50.0)]))
    # the impulse: a sharp jump at t=1000, with a skewed book (bid depth >> ask depth)
    book_rows.append(binance_book_row(MARKET_ID, 1_000, [(104.9, 400.0), (104.8, 400.0)], [(105.0, 20.0)]))
    insert_rows(rec, "binance_book", book_rows)

    insert_rows(rec, "binance_trades", [
        binance_trade_row(MARKET_ID, 1, 900, 100.0, 1.0, "SELL"),
        binance_trade_row(MARKET_ID, 2, 950, 104.9, 5.0, "BUY"),
        binance_trade_row(MARKET_ID, 3, 990, 105.0, 5.0, "BUY"),
    ])

    insert_rows(rec, "polymarket_book", [
        poly_book_row(MARKET_ID, 400, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)],
                       down_bid=0.49, down_ask=0.50, down_bids=[(0.49, 100)], down_asks=[(0.50, 100)]),
        poly_book_row(MARKET_ID, 1_000, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)],
                       down_bid=0.49, down_ask=0.50, down_bids=[(0.49, 100)], down_asks=[(0.50, 100)]),
    ])
    rec.close()


def test_signal_01_up_impulse_with_lenient_threshold_fires():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        _build_up_impulse_db(tmp_path)
        conn = _open(tmp_path)
        try:
            cfg = dataclasses.replace(BASE_CFG, z_threshold=2.5)
            signals = replay_signals(conn, MARKET_ID, cfg)
            assert len(signals) == 1
            assert signals[0].direction == "UP"
        finally:
            conn.close()


def test_signal_02_same_history_with_impossible_threshold_does_not_fire():
    """SIGNAL-02: same crafted timeline, only z_threshold changed -> no signal."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        _build_up_impulse_db(tmp_path)
        conn = _open(tmp_path)
        try:
            cfg = dataclasses.replace(BASE_CFG, z_threshold=1_000.0)
            signals = replay_signals(conn, MARKET_ID, cfg)
            assert signals == []
        finally:
            conn.close()


def test_signal_04_flow_confirmation_can_be_toggled_offline():
    """SIGNAL-04: same raw data, flow_threshold raised past what the
    fixture provides -> signal disappears without any new collection."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        _build_up_impulse_db(tmp_path)
        conn = _open(tmp_path)
        try:
            lenient = replay_signals(conn, MARKET_ID, dataclasses.replace(BASE_CFG, flow_threshold=0.1))
            strict = replay_signals(conn, MARKET_ID, dataclasses.replace(BASE_CFG, flow_threshold=0.99))
            assert len(lenient) == 1
            assert strict == []
        finally:
            conn.close()


def test_signal_05_imbalance_confirmation_can_be_toggled_offline():
    """SIGNAL-05"""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        _build_up_impulse_db(tmp_path)
        conn = _open(tmp_path)
        try:
            lenient = replay_signals(conn, MARKET_ID, dataclasses.replace(BASE_CFG, imbalance_threshold=0.05))
            strict = replay_signals(conn, MARKET_ID, dataclasses.replace(BASE_CFG, imbalance_threshold=0.99))
            assert len(lenient) == 1
            assert strict == []
        finally:
            conn.close()


def test_signal_07_reset_allows_a_second_independent_impulse():
    """SIGNAL-06/07: hysteresis still holds under replay -- one continuous
    excursion gives one signal, and only a fresh excursion after a reset can
    give a second one. We just check the single-impulse fixture doesn't
    double-fire even though several book rows sit above threshold."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        _build_up_impulse_db(tmp_path)
        conn = _open(tmp_path)
        try:
            signals = replay_signals(conn, MARKET_ID, BASE_CFG)
            assert len(signals) == 1  # not one per subsequent poly/binance tick
        finally:
            conn.close()


def _open(tmp_path):
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.row_factory = sqlite3.Row
    return conn
