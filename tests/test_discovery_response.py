"""Execution marks (no-lookahead entry pricing) and future-response
analysis (executable return, MFE/MAE, first-passage, opposite-side hedge).
"""
from poly_analyzer.discovery import (
    analyze_response, compute_execution_mark, opposite_side_lock_path,
)
from tests.discovery_fixtures import insert_rows, make_recorder, poly_book_row

MARKET_ID = "M1"


def _db(tmp_path, rows):
    rec = make_recorder(tmp_path)
    insert_rows(rec, "polymarket_book", rows)
    rec.close()
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.row_factory = sqlite3.Row
    return conn


def test_exec_01_02_uses_first_state_after_target_not_before(tmp_path):
    """EXEC-01/02: signal at t=0, latency=250ms -> target=250. Book states
    exist at t=240 (before target, must be ignored) and t=263 (after
    target, must be used)."""
    conn = _db(tmp_path, [
        poly_book_row(MARKET_ID, 240, up_ask=0.53, up_asks=[(0.53, 100)]),
        poly_book_row(MARKET_ID, 263, up_ask=0.55, up_asks=[(0.55, 100)]),
    ])
    try:
        mark = compute_execution_mark(conn, MARKET_ID, "UP", signal_ts=0, latency_ms=250, size=100)
        assert mark.actual_ts == 263
        assert mark.entry_vwap == 0.55
    finally:
        conn.close()


def test_exec_03_no_liquidity_at_execution_time_is_not_executable(tmp_path):
    conn = _db(tmp_path, [
        poly_book_row(MARKET_ID, 300, up_ask=None, up_asks=[]),
    ])
    try:
        mark = compute_execution_mark(conn, MARKET_ID, "UP", signal_ts=0, latency_ms=250, size=100)
        assert mark.entry_vwap is None
    finally:
        conn.close()


def test_exec_04_latency_variants_are_independent(tmp_path):
    conn = _db(tmp_path, [
        poly_book_row(MARKET_ID, 100, up_ask=0.52, up_asks=[(0.52, 100)]),
        poly_book_row(MARKET_ID, 250, up_ask=0.54, up_asks=[(0.54, 100)]),
        poly_book_row(MARKET_ID, 500, up_ask=0.57, up_asks=[(0.57, 100)]),
    ])
    try:
        m100 = compute_execution_mark(conn, MARKET_ID, "UP", signal_ts=0, latency_ms=100, size=100)
        m250 = compute_execution_mark(conn, MARKET_ID, "UP", signal_ts=0, latency_ms=250, size=100)
        m500 = compute_execution_mark(conn, MARKET_ID, "UP", signal_ts=0, latency_ms=500, size=100)
        assert (m100.entry_vwap, m250.entry_vwap, m500.entry_vwap) == (0.52, 0.54, 0.57)
    finally:
        conn.close()


def test_resp_01_executable_return_is_future_sell_vwap_minus_entry_vwap(tmp_path):
    conn = _db(tmp_path, [
        poly_book_row(MARKET_ID, 1_000, up_bid=0.60, up_bids=[(0.60, 100)]),
    ])
    try:
        stats = analyze_response(conn, MARKET_ID, "UP", entry_ts=0, entry_vwap=0.52, size=100)
        assert abs(stats.final_return - (0.60 - 0.52)) < 1e-9
    finally:
        conn.close()


def test_resp_02_spike_with_insufficient_size_does_not_inflate_return(tmp_path):
    """A best-bid spike to .70 with only 10 shares of depth must not look
    like a +.18 return for a 100-share exit -- the VWAP blends in the
    deeper, worse-priced levels."""
    conn = _db(tmp_path, [
        poly_book_row(MARKET_ID, 1_000, up_bid=0.70, up_bids=[(0.70, 10.0), (0.50, 200.0)]),
    ])
    try:
        stats = analyze_response(conn, MARKET_ID, "UP", entry_ts=0, entry_vwap=0.52, size=100)
        expected_vwap = (10 * 0.70 + 90 * 0.50) / 100
        assert abs(stats.final_return - (expected_vwap - 0.52)) < 1e-9
        assert stats.final_return < 0.70 - 0.52  # nowhere near the naive best-bid return
    finally:
        conn.close()


def test_resp_03_04_mfe_mae_relative_to_entry(tmp_path):
    conn = _db(tmp_path, [
        poly_book_row(MARKET_ID, 100, up_bid=0.50, up_bids=[(0.50, 100)]),
        poly_book_row(MARKET_ID, 200, up_bid=0.48, up_bids=[(0.48, 100)]),
        poly_book_row(MARKET_ID, 300, up_bid=0.49, up_bids=[(0.49, 100)]),
        poly_book_row(MARKET_ID, 400, up_bid=0.55, up_bids=[(0.55, 100)]),
        poly_book_row(MARKET_ID, 500, up_bid=0.53, up_bids=[(0.53, 100)]),
    ])
    try:
        stats = analyze_response(conn, MARKET_ID, "UP", entry_ts=0, entry_vwap=0.50, size=100)
        assert abs(stats.mfe - 0.05) < 1e-9  # .55 - .50
        assert abs(stats.mae - (-0.02)) < 1e-9  # .48 - .50
        assert stats.time_to_mfe_ms == 400
        assert stats.time_to_mae_ms == 200
    finally:
        conn.close()


def test_resp_05_time_to_plus_is_first_touch_only(tmp_path):
    conn = _db(tmp_path, [
        poly_book_row(MARKET_ID, 100, up_bid=0.51, up_bids=[(0.51, 100)]),  # +.01
        poly_book_row(MARKET_ID, 200, up_bid=0.53, up_bids=[(0.53, 100)]),  # +.03
        poly_book_row(MARKET_ID, 300, up_bid=0.52, up_bids=[(0.52, 100)]),  # back to +.02, must not move time_to_+.01
    ])
    try:
        stats = analyze_response(conn, MARKET_ID, "UP", entry_ts=0, entry_vwap=0.50, size=100,
                                  levels_usd=(0.01, 0.02, 0.03))
        assert stats.time_to_plus_ms[0.01] == 100
        assert stats.time_to_plus_ms[0.02] == 200  # first time return>=.02 was at t=200 (+.03)
        assert stats.time_to_plus_ms[0.03] == 200
    finally:
        conn.close()


def test_hedge_01_combined_cost_is_entry_plus_opposite_ask_vwap(tmp_path):
    conn = _db(tmp_path, [
        poly_book_row(MARKET_ID, 1_000, down_ask=0.44, down_asks=[(0.44, 100)]),
    ])
    try:
        path = opposite_side_lock_path(conn, MARKET_ID, "UP", entry_ts=0, entry_vwap=0.52, size=100)
        assert len(path) == 1
        assert abs(path[0].combined_cost - 0.96) < 1e-9
    finally:
        conn.close()


def test_hedge_02_uses_vwap_not_best_ask_for_the_opposite_side(tmp_path):
    conn = _db(tmp_path, [
        poly_book_row(MARKET_ID, 1_000, down_ask=0.44, down_asks=[(0.44, 10.0), (0.50, 200.0)]),
    ])
    try:
        path = opposite_side_lock_path(conn, MARKET_ID, "UP", entry_ts=0, entry_vwap=0.52, size=100)
        expected_opposite_vwap = (10 * 0.44 + 90 * 0.50) / 100
        assert abs(path[0].opposite_ask_vwap - expected_opposite_vwap) < 1e-9
    finally:
        conn.close()


def test_hedge_03_unavailable_when_opposite_liquidity_is_zero(tmp_path):
    conn = _db(tmp_path, [
        poly_book_row(MARKET_ID, 1_000, down_ask=None, down_asks=[]),
    ])
    try:
        path = opposite_side_lock_path(conn, MARKET_ID, "UP", entry_ts=0, entry_vwap=0.52, size=100)
        assert path[0].combined_cost is None
    finally:
        conn.close()
