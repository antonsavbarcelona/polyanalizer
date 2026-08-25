"""Golden end-to-end fixture (test-cases.md section O): one deterministic
timeline, replayed through signal detection -> execution mark -> response
analysis as one pipeline. Regression-pins the composed result so a future
change to any piece is caught immediately if it silently changes the
mechanics.
"""
import sqlite3

from poly_analyzer.discovery import (
    DiscoverySignalConfig, analyze_response, compute_execution_mark, replay_signals,
)
from tests.discovery_fixtures import (
    binance_book_row, binance_trade_row, insert_rows, make_recorder, market_row, poly_book_row,
)

MARKET_ID = "GOLDEN"

CFG = DiscoverySignalConfig(
    momentum_window_ms=100, volatility_lookback_ms=1_000, sigma_min_samples=5,
    flow_window_ms=500, flow_threshold=0.3,
    imbalance_depth=5, imbalance_threshold=0.10,
    poly_lag_window_ms=500, max_poly_reprice=0.01,
    tte_min_s=0.0, tte_max_s=1_000_000.0,
    ask_min=0.20, ask_max=0.80, spread_max=0.03,
)


def _build_golden_db(tmp_path):
    rec = make_recorder(tmp_path)
    insert_rows(rec, "markets", [market_row(MARKET_ID, end_ts=500_000)])

    # 0-900ms: flat, quiet market. 1000ms: sharp UP impulse (same shape as
    # test_discovery_signals' fixture, known to trigger a UP signal there).
    book_rows = [binance_book_row(MARKET_ID, ts, [(100.0, 50.0)], [(100.02, 50.0)])
                 for ts in range(0, 1_000, 100)]
    book_rows.append(binance_book_row(MARKET_ID, 1_000, [(104.9, 400.0), (104.8, 400.0)], [(105.0, 20.0)]))
    insert_rows(rec, "binance_book", book_rows)
    insert_rows(rec, "binance_trades", [
        binance_trade_row(MARKET_ID, 1, 900, 100.0, 1.0, "SELL"),
        binance_trade_row(MARKET_ID, 2, 950, 104.9, 5.0, "BUY"),
        binance_trade_row(MARKET_ID, 3, 990, 105.0, 5.0, "BUY"),
    ])

    # Polymarket UP: stable at .50 through the signal, then reprices upward
    # over the following second -- exactly the repricing this hypothesis
    # is trying to detect early.
    insert_rows(rec, "polymarket_book", [
        poly_book_row(MARKET_ID, 400, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)],
                       down_bid=0.49, down_ask=0.50, down_bids=[(0.49, 100)], down_asks=[(0.50, 100)]),
        poly_book_row(MARKET_ID, 1_000, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)],
                       down_bid=0.49, down_ask=0.50, down_bids=[(0.49, 100)], down_asks=[(0.50, 100)]),
        poly_book_row(MARKET_ID, 1_150, up_bid=0.50, up_ask=0.52, up_bids=[(0.50, 100)], up_asks=[(0.52, 100)]),
        poly_book_row(MARKET_ID, 1_300, up_bid=0.53, up_ask=0.55, up_bids=[(0.53, 100)], up_asks=[(0.55, 100)]),
        poly_book_row(MARKET_ID, 1_600, up_bid=0.56, up_ask=0.58, up_bids=[(0.56, 100)], up_asks=[(0.58, 100)]),
        poly_book_row(MARKET_ID, 2_000, up_bid=0.55, up_ask=0.57, up_bids=[(0.55, 100)], up_asks=[(0.57, 100)]),
    ])
    rec.close()


def test_golden_pipeline_signal_to_execution_to_response(tmp_path):
    _build_golden_db(tmp_path)
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.row_factory = sqlite3.Row
    try:
        signals = replay_signals(conn, MARKET_ID, CFG)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == "UP"
        assert sig.signal_ts == 1_000
        assert sig.poly_ask == 0.50  # not yet repriced at signal time

        mark = compute_execution_mark(conn, MARKET_ID, "UP", sig.signal_ts, latency_ms=250, size=100)
        assert mark.actual_ts == 1_300  # first state at/after 1000+250=1250
        assert mark.entry_vwap == 0.55

        stats = analyze_response(conn, MARKET_ID, "UP", entry_ts=mark.actual_ts,
                                  entry_vwap=mark.entry_vwap, size=100)
        # Poly kept rising to .56 bid (t=1600) before settling back to .55 (t=2000).
        assert abs(stats.mfe - (0.56 - 0.55)) < 1e-9
        assert stats.time_to_mfe_ms == 300  # 1600 - 1300
        assert stats.final_return == 0.0  # .55 bid at t=2000 vs .55 entry
    finally:
        conn.close()
