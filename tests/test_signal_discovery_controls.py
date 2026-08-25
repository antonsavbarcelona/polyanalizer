"""Deterministic matched controls (IMPLEMENTATION CONTRACT #24-27): same
inputs must always pick the same K controls, in the same order, with no
Python `random` involved anywhere."""
from __future__ import annotations

from tests.discovery_fixtures import binance_book_row, insert_rows, make_recorder, market_row, poly_book_row

from research.discovery.controls import select_matched_controls
from research.discovery_types import SignalDiscoveryConfig

MARKET_END_TS = 10_000_000
SOURCE_TS = 30_000
SOURCE_TTE_S = (MARKET_END_TS - SOURCE_TS) / 1000.0  # 9_970.0


def _build_flat_market(tmp_path):
    """A 60s-long market with a perfectly flat Binance price (momentum==0
    everywhere -> nothing ever self-triggers the config below) and a
    constant Polymarket UP book (same price/spread regime at every tick).
    Only time-to-expiry drifts (it must -- time passes), so match_distance
    is NOT flat across candidates: it's smallest right at the edges of the
    +/-10s exclusion window around SOURCE_TS and grows with distance from
    it. That's exactly the ranking contract#25 asks for, and is what's
    verified below directly (not by re-deriving the exact numbers)."""
    from poly_analyzer.discovery import open_db

    rec = make_recorder(tmp_path)
    insert_rows(rec, "markets", [market_row("M_CTRL", end_ts=MARKET_END_TS)])
    insert_rows(rec, "binance_book", [
        binance_book_row("M_CTRL", ts, [(100.0, 50.0)], [(100.02, 50.0)]) for ts in range(0, 60_001, 500)
    ])
    # up_bid=0.485/up_ask=0.50 -> spread=0.015 (constant "1-2c" bucket) at every 2s tick.
    insert_rows(rec, "polymarket_book", [
        poly_book_row("M_CTRL", ts, up_bid=0.485, up_ask=0.50, up_bids=[(0.485, 100)], up_asks=[(0.50, 100)])
        for ts in range(0, 60_001, 2_000)
    ])
    rec.close()
    return open_db(str(tmp_path / "test.db"))


def _select(conn, k=5):
    cfg = SignalDiscoveryConfig(id="C_CTRL", momentum_window_ms=1_000, absolute_move_threshold_bps=50.0)
    return select_matched_controls(
        conn, market_id="M_CTRL", asset="BTC", cfg=cfg, source_signal_id="SRC",
        direction="UP", source_tte_s=SOURCE_TTE_S, source_price=0.50, source_spread=0.015,
        source_vol_30s=0.0, source_tte_regime="600s+", source_price_regime="0.50-0.65",
        source_spread_regime="1-2c", source_volatility_regime=None,
        all_signal_ts_this_config=[SOURCE_TS], vol_boundaries=None, k=k,
    )


def test_controls_are_deterministic(tmp_path):
    conn = _build_flat_market(tmp_path)

    controls_1 = _select(conn)
    controls_2 = _select(conn)

    assert [c.control_id for c in controls_1] == [c.control_id for c in controls_2]
    assert [c.control_ts for c in controls_1] == [c.control_ts for c in controls_2]
    assert [c.match_distance for c in controls_1] == [c.match_distance for c in controls_2]


def test_controls_respect_exclusion_window_and_regimes(tmp_path):
    conn = _build_flat_market(tmp_path)
    controls = _select(conn, k=5)

    assert len(controls) == 5
    assert [c.match_rank for c in controls] == [1, 2, 3, 4, 5]

    for c in controls:
        assert abs(c.control_ts - SOURCE_TS) > 10_000  # never inside the +/-10s exclusion window
        assert c.tte_regime == "600s+"
        assert c.price_regime == "0.50-0.65"
        assert c.spread_regime == "1-2c"
        assert c.source_signal_id == "SRC"
        assert c.signal_config_id == "C_CTRL"
        assert c.direction == "UP"

    # contract #25: match_distance ASC, timestamp ASC on ties.
    for a, b in zip(controls, controls[1:]):
        assert a.match_distance <= b.match_distance + 1e-12
        if abs(a.match_distance - b.match_distance) < 1e-12:
            assert a.control_ts < b.control_ts

    # Price/spread/vol components are all exactly 0 by construction (flat
    # market), so match_distance is driven purely by |TTE drift| -- which
    # is minimized right at the two edges of the exclusion window, meaning
    # the two closest picks must straddle it symmetrically.
    ts_values = sorted(c.control_ts for c in controls[:2])
    assert ts_values[0] == SOURCE_TS - 10_500  # 19_500: nearest eligible tick before the window
    assert ts_values[1] == SOURCE_TS + 10_500  # 40_500: nearest eligible tick after the window


def test_controls_respect_k_limit(tmp_path):
    conn = _build_flat_market(tmp_path)
    controls = _select(conn, k=3)
    assert len(controls) == 3
    for c in controls:
        assert abs(c.control_ts - SOURCE_TS) > 10_000
