import sqlite3

import pytest

from poly_analyzer.discovery import DiscoveredSignal, ExecutionMark
from poly_analyzer.db import SCHEMA
from research.execution.episode import SignalEpisode, build_episode
from research.execution.exit_policy import evaluate_exit
from research.fees import FeeModel
from research.types import ExitConfig
from tests.discovery_fixtures import poly_book_row

MARKET_ID = "M_RESEARCH"
MARKET = {"asset": "BTC", "fee_rate": 0.07, "fee_exponent": 1.0, "end_ts": 10_000}
SIZE = 100.0


def _signal(signal_ts=900, direction="UP"):
    return DiscoveredSignal(
        config_id="cfg",
        market_id=MARKET_ID,
        direction=direction,
        signal_ts=signal_ts,
        z=3.0,
        momentum=0.01,
        flow=0.5,
        imbalance=0.2,
        poly_ask=0.53,
        poly_bid=0.52,
        poly_spread=0.01,
        remaining_s=300.0,
        ask_change=0.0,
    )


def _episode(rows, entry_vwap=0.53, entry_ts=1_000):
    mark = ExecutionMark(
        latency_ms=100,
        target_ts=entry_ts,
        actual_ts=entry_ts,
        entry_vwap=entry_vwap,
        filled_size=SIZE,
        best_ask=entry_vwap,
    )
    return SignalEpisode(
        signal=_signal(signal_ts=entry_ts - 100),
        execution_config_id="lat100_size100",
        entry_mark=mark,
        path_rows=rows,
        path_ts=[row["ts"] for row in rows],
        market_row=dict(MARKET),
    )


def _insert(conn, table, row):
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{key}" for key in row.keys())
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", row)


def test_build_episode_rejects_partial_entry_fill():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _insert(conn, "polymarket_book", poly_book_row(
        MARKET_ID, 1_000, up_ask=0.53, up_asks=[(0.53, 50.0)]
    ))
    try:
        episode = build_episode(conn, _signal(signal_ts=900), "lat100_size100", 100, SIZE, MARKET)
        assert episode is None
    finally:
        conn.close()


def test_maker_tp_fills_at_target_price_and_pays_only_entry_fee():
    episode = _episode([
        poly_book_row(MARKET_ID, 1_000, up_bid=0.52, up_bids=[(0.52, SIZE)]),
        poly_book_row(MARKET_ID, 1_800, up_bid=0.57, up_bids=[(0.57, SIZE)]),
    ])
    cfg = ExitConfig.absolute_delta("tp4_sl3_hold10s", tp_delta=0.04, sl_delta=0.03, max_holding_ms=10_000)

    result = evaluate_exit(episode, cfg, FeeModel(), SIZE)

    assert result.exit_reason == "TP"
    assert result.exit_ts == 1_800
    assert result.exit_price == pytest.approx(0.57)
    assert result.tp_hit is True
    assert result.fees_per_share == pytest.approx(FeeModel().taker_fee_per_share(0.53, MARKET))
    assert result.net_pnl_per_share == pytest.approx(0.04 - result.fees_per_share)


def test_ask_touching_target_is_not_a_maker_tp_fill():
    episode = _episode([
        poly_book_row(MARKET_ID, 1_100, up_bid=0.56, up_ask=0.57, up_bids=[(0.56, SIZE)]),
        poly_book_row(MARKET_ID, 2_000, up_bid=0.56, up_bids=[(0.56, SIZE)]),
    ])
    cfg = ExitConfig.absolute_delta("tp4_hold1s", tp_delta=0.04, max_holding_ms=1_000)

    result = evaluate_exit(episode, cfg, FeeModel(), SIZE)

    assert result.exit_reason == "TIMEOUT"
    assert result.tp_hit is False
    assert result.exit_ts == 2_000
    assert result.exit_price == pytest.approx(0.56)


def test_stop_loss_executes_after_configured_latency():
    episode = _episode([
        poly_book_row(MARKET_ID, 1_100, up_bid=0.49, up_bids=[(0.49, SIZE)]),
        poly_book_row(MARKET_ID, 1_250, up_bid=0.48, up_bids=[(0.48, SIZE)]),
        poly_book_row(MARKET_ID, 1_300, up_bid=0.47, up_bids=[(0.47, SIZE)]),
    ])
    cfg = ExitConfig.absolute_delta(
        "sl3_latency200",
        sl_delta=0.03,
        max_holding_ms=10_000,
        stop_exit_latency_ms=200,
    )

    result = evaluate_exit(episode, cfg, FeeModel(), SIZE)

    assert result.exit_reason == "SL"
    assert result.sl_hit is True
    assert result.exit_ts == 1_300
    assert result.exit_price == pytest.approx(0.47)


def test_ambiguous_tp_and_sl_same_row_uses_sl_worst_case():
    episode = _episode([
        poly_book_row(MARKET_ID, 1_100, up_bid=0.58, up_bids=[(0.58, 10.0), (0.49, 90.0)]),
    ])
    cfg = ExitConfig.absolute_delta("ambiguous", tp_delta=0.04, sl_delta=0.03, max_holding_ms=10_000)

    result = evaluate_exit(episode, cfg, FeeModel(), SIZE)

    assert result.exit_reason == "SL"
    assert result.ambiguous_exit is True
    assert result.sl_hit is True
    assert result.tp_hit is False
    assert result.exit_price == pytest.approx((0.58 * 10.0 + 0.49 * 90.0) / SIZE)


def test_timeout_priority_over_tp_seen_only_after_timeout_boundary():
    episode = _episode([
        poly_book_row(MARKET_ID, 2_001, up_bid=0.60, up_bids=[(0.60, SIZE)]),
    ])
    cfg = ExitConfig.absolute_delta("tp4_hold1s", tp_delta=0.04, max_holding_ms=1_000)

    result = evaluate_exit(episode, cfg, FeeModel(), SIZE)

    assert result.exit_reason == "TIMEOUT"
    assert result.tp_hit is False
    assert result.exit_ts == 2_001


def test_timeout_skips_rows_with_insufficient_exit_liquidity():
    episode = _episode([
        poly_book_row(MARKET_ID, 2_000, up_bid=0.57, up_bids=[(0.57, 50.0)]),
        poly_book_row(MARKET_ID, 2_100, up_bid=0.56, up_bids=[(0.56, SIZE)]),
    ])
    cfg = ExitConfig.absolute_delta("hold1s", max_holding_ms=1_000)

    result = evaluate_exit(episode, cfg, FeeModel(), SIZE)

    assert result.exit_reason == "TIMEOUT"
    assert result.exit_ts == 2_100
    assert result.exit_price == pytest.approx(0.56)
