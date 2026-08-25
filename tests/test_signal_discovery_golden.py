"""Golden deterministic fixture for the signal-response discovery pipeline
(IMPLEMENTATION CONTRACT #42-44). Every expected value below is a literal,
hand-computed constant -- never derived by calling the production code --
so this test can only pass if compute_entry_mark/compute_signal_response
actually implement the contract's arithmetic, not just "whatever the code
currently does".

Four signals, each isolating one contract requirement:
  A (market M_A, UP)   -- profitable multi-level entry AND exit VWAP.
  B (market M_B, DOWN) -- mirrors A on the DOWN side/token.
  C (market M_C, UP)   -- insufficient ASK depth at entry -> NOT_EXECUTABLE.
  D (market M_D, UP)   -- entry fills fine, but future BID depth is
                          insufficient -> NOT_SELL_EXECUTABLE (never a fake
                          zero/loss).
"""
from __future__ import annotations

from tests.discovery_fixtures import insert_rows, make_recorder, market_row, poly_book_row

from research.discovery.entry import compute_entry_mark
from research.discovery.response import compute_signal_response
from research.discovery_types import SignalSnapshot
from research.fees import FeeModel


def _snapshot(signal_id, market_id, direction, signal_ts) -> SignalSnapshot:
    """Minimal SignalSnapshot: only the fields compute_entry_mark/
    compute_signal_response actually read (signal_id, signal_config_id,
    asset, market_id, direction, signal_ts). Everything else is irrelevant
    to this layer (it's produced by research.discovery.detect, tested
    separately) so it's left None/0 -- SR-E2E golden coverage is scoped to
    entry+response arithmetic, not the detection trigger."""
    return SignalSnapshot(
        signal_id=signal_id, signal_config_id="C_GOLDEN", asset="BTC", market_id=market_id,
        direction=direction, signal_ts=signal_ts, binance_mid=None,
        return_100ms=None, return_250ms=None, return_500ms=None, return_750ms=None, return_1s=None,
        return_1_5s=None, return_2s=None, return_3s=None, return_5s=None, return_10s=None,
        active_momentum_value=None, vol_10s=None, vol_30s=None, vol_60s=None, vol_120s=None, vol_300s=None,
        active_volatility=None, z_score=None,
        flow_100ms=None, flow_250ms=None, flow_500ms=None, flow_750ms=None, flow_1s=None, flow_2s=None,
        flow_3s=None, flow_5s=None, flow_10s=None, active_flow=None, volume_z=None,
        imbalance_top1=None, imbalance_top3=None, imbalance_top5=None, imbalance_top10=None,
        microprice_bias=None, target_bid=None, target_ask=None, target_spread=None,
        target_bid_size=None, target_ask_size=None,
        poly_move_100ms=None, poly_move_250ms=None, poly_move_500ms=None, poly_move_1s=None, poly_move_2s=None,
        time_remaining_ms=None, time_elapsed_ms=None, reference_price=None, chainlink_twap=None,
        distance_to_reference_bps=None, volatility_regime=None, tte_regime="120-300s",
        price_regime=None, spread_regime=None,
    )


MARKET_ROW = {"end_ts": 100_000, "fee_rate": 0.07, "fee_exponent": 1.0}


def _build_golden_db(tmp_path):
    rec = make_recorder(tmp_path)

    # --- Signal A: UP, profitable, multi-level entry AND exit VWAP ---
    insert_rows(rec, "markets", [market_row("M_A", end_ts=100_000)])
    insert_rows(rec, "polymarket_book", [
        poly_book_row("M_A", 1_000, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 20)]),
        poly_book_row("M_A", 1_300, up_bid=0.49, up_ask=0.51,
                       up_bids=[(0.49, 100)], up_asks=[(0.51, 40), (0.52, 60)]),
        poly_book_row("M_A", 2_300, up_bid=0.55, up_ask=0.60,
                       up_bids=[(0.55, 50), (0.54, 60)], up_asks=[(0.60, 100)]),
    ])

    # --- Signal B: DOWN, mirrors A ---
    insert_rows(rec, "markets", [market_row("M_B", end_ts=100_000)])
    insert_rows(rec, "polymarket_book", [
        poly_book_row("M_B", 1_300, down_bid=0.40, down_ask=0.48,
                       down_bids=[(0.40, 100)], down_asks=[(0.48, 30), (0.49, 70)]),
        poly_book_row("M_B", 2_300, down_bid=0.51, down_ask=0.60,
                       down_bids=[(0.51, 60), (0.50, 50)], down_asks=[(0.60, 100)]),
    ])

    # --- Signal C: UP, insufficient ask depth at entry ---
    insert_rows(rec, "markets", [market_row("M_C", end_ts=100_000)])
    insert_rows(rec, "polymarket_book", [
        poly_book_row("M_C", 1_300, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 20)]),
    ])

    # --- Signal D: UP, entry fills, future bid depth insufficient ---
    insert_rows(rec, "markets", [market_row("M_D", end_ts=100_000)])
    insert_rows(rec, "polymarket_book", [
        poly_book_row("M_D", 1_300, up_bid=0.49, up_ask=0.50,
                       up_bids=[(0.49, 100)], up_asks=[(0.50, 50), (0.51, 50)]),
        poly_book_row("M_D", 2_300, up_bid=0.52, up_ask=0.60, up_bids=[(0.52, 30)], up_asks=[(0.60, 100)]),
    ])

    rec.close()
    from poly_analyzer.discovery import open_db
    return open_db(str(tmp_path / "test.db"))


def test_golden_signal_A_up_profitable(tmp_path):
    conn = _build_golden_db(tmp_path)
    signal = _snapshot("SIG_A", "M_A", "UP", signal_ts=1_000)

    entry = compute_entry_mark(conn, signal, latency_ms=250, size_shares=100, market_row=MARKET_ROW,
                                fee_model=FeeModel())

    assert entry.status == "EXECUTED"
    assert entry.entry_target_ts == 1_250
    assert entry.entry_actual_ts == 1_300  # first VALID state >= 1250, not the 1_000 row before target
    assert entry.entry_delay_after_target_ms == 50
    assert entry.entry_best_ask == 0.51
    assert entry.entry_vwap == (0.51 * 40 + 0.52 * 60) / 100  # == 0.516
    assert abs(entry.entry_vwap - 0.516) < 1e-9
    assert abs(entry.entry_slippage - 0.006) < 1e-9
    assert abs(entry.entry_fee_per_share - 0.07 * 0.484 / 2.0) < 1e-9  # 0.01694
    assert abs(entry.entry_fee_total - entry.entry_fee_per_share * 100) < 1e-9

    response = compute_signal_response(conn, signal, entry, horizon_ms=1_000, market_row=MARKET_ROW)
    assert response.status == "AVAILABLE"
    assert response.response_target_ts == 1_300 + 1_000  # measured from ACTUAL entry, not signal_ts
    assert response.response_actual_ts == 2_300
    assert response.future_best_bid == 0.55
    assert abs(response.future_sell_vwap - (50 * 0.55 + 50 * 0.54) / 100) < 1e-9  # == 0.545
    assert abs(response.raw_response - (0.545 - 0.516)) < 1e-9  # == 0.029
    assert abs(response.fee_adjusted_response - (0.029 - entry.entry_fee_per_share)) < 1e-9
    assert response.response_positive is True
    assert response.fee_adjusted_positive is True


def test_golden_signal_B_down_mirrors_A(tmp_path):
    conn = _build_golden_db(tmp_path)
    signal = _snapshot("SIG_B", "M_B", "DOWN", signal_ts=1_000)

    entry = compute_entry_mark(conn, signal, latency_ms=250, size_shares=100, market_row=MARKET_ROW,
                                fee_model=FeeModel())
    assert entry.status == "EXECUTED"
    assert entry.entry_best_ask == 0.48
    assert abs(entry.entry_vwap - (0.48 * 30 + 0.49 * 70) / 100) < 1e-9  # == 0.487
    assert abs(entry.entry_slippage - 0.007) < 1e-9

    response = compute_signal_response(conn, signal, entry, horizon_ms=1_000, market_row=MARKET_ROW)
    assert response.status == "AVAILABLE"
    assert abs(response.future_sell_vwap - (60 * 0.51 + 40 * 0.50) / 100) < 1e-9  # == 0.506
    assert abs(response.raw_response - (0.506 - 0.487)) < 1e-9  # == 0.019
    assert response.response_positive is True


def test_golden_signal_C_insufficient_entry_liquidity(tmp_path):
    conn = _build_golden_db(tmp_path)
    signal = _snapshot("SIG_C", "M_C", "UP", signal_ts=1_000)

    entry = compute_entry_mark(conn, signal, latency_ms=250, size_shares=100, market_row=MARKET_ROW,
                                fee_model=FeeModel())
    assert entry.status == "NOT_EXECUTABLE"
    assert entry.entry_vwap is None
    assert entry.entry_fee_per_share is None
    assert entry.available_ask_liquidity == 20
    # entry_actual_ts/best_ask are still recorded -- we DID find a valid
    # book state, it just couldn't fill the full requested size.
    assert entry.entry_actual_ts == 1_300
    assert entry.entry_best_ask == 0.50


def test_golden_signal_D_insufficient_future_sell_liquidity(tmp_path):
    conn = _build_golden_db(tmp_path)
    signal = _snapshot("SIG_D", "M_D", "UP", signal_ts=1_000)

    entry = compute_entry_mark(conn, signal, latency_ms=250, size_shares=100, market_row=MARKET_ROW,
                                fee_model=FeeModel())
    assert entry.status == "EXECUTED"
    assert abs(entry.entry_vwap - (0.50 * 50 + 0.51 * 50) / 100) < 1e-9  # == 0.505

    response = compute_signal_response(conn, signal, entry, horizon_ms=1_000, market_row=MARKET_ROW)
    assert response.status == "NOT_SELL_EXECUTABLE"
    assert response.future_sell_vwap is None
    assert response.raw_response is None
    assert response.fee_adjusted_response is None
    # NULL, never a fake zero/loss (contract #10, #13):
    assert response.response_positive is None
    assert response.fee_adjusted_positive is None
    assert response.available_bid_liquidity == 30
    assert response.future_best_bid == 0.52


def test_golden_response_after_market_end(tmp_path):
    conn = _build_golden_db(tmp_path)
    signal = _snapshot("SIG_A", "M_A", "UP", signal_ts=1_000)

    entry = compute_entry_mark(conn, signal, latency_ms=250, size_shares=100, market_row=MARKET_ROW,
                                fee_model=FeeModel())
    assert entry.entry_actual_ts == 1_300

    # entry_actual_ts(1300) + horizon(2000) = 3300, past a market that ends at 2500.
    tight_market_row = {"end_ts": 2_500, "fee_rate": 0.07, "fee_exponent": 1.0}
    response = compute_signal_response(conn, signal, entry, horizon_ms=2_000, market_row=tight_market_row)
    assert response.status == "AFTER_MARKET_END"
    assert response.response_actual_ts is None
    assert response.raw_response is None
    assert response.response_positive is None


def test_golden_response_no_data(tmp_path):
    conn = _build_golden_db(tmp_path)
    signal = _snapshot("SIG_A", "M_A", "UP", signal_ts=1_000)

    entry = compute_entry_mark(conn, signal, latency_ms=250, size_shares=100, market_row=MARKET_ROW,
                                fee_model=FeeModel())
    # Horizon so large the target timestamp is past every recorded book row
    # for M_A (last row at ts=2300), but still within a generous market_end.
    response = compute_signal_response(conn, signal, entry, horizon_ms=50_000, market_row=MARKET_ROW)
    assert response.status == "NO_DATA"
    assert response.response_actual_ts is None
    assert response.raw_response is None


def test_golden_entry_no_data(tmp_path):
    conn = _build_golden_db(tmp_path)
    # signal_ts far past every recorded row for M_A -> no valid state at/after target.
    signal = _snapshot("SIG_LATE", "M_A", "UP", signal_ts=90_000)

    entry = compute_entry_mark(conn, signal, latency_ms=250, size_shares=100, market_row=MARKET_ROW,
                                fee_model=FeeModel())
    assert entry.status == "NO_DATA"
    assert entry.entry_actual_ts is None
    assert entry.entry_vwap is None
