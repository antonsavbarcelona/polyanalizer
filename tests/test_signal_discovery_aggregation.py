"""Summary-table aggregation (IMPLEMENTATION CONTRACT #29-33), hand-computed
over small in-memory fixtures -- no DB involved, these are pure functions."""
from __future__ import annotations

from research.discovery.aggregation import (
    compute_first_passage_summary,
    compute_hit_summary,
    compute_signal_config_summary,
    compute_signal_response_summary,
)
from research.discovery_types import ControlResponse, DiscoveryEntryMark, SignalPathStats, SignalResponse


def _entry(mid: str, status: str, slippage=None) -> DiscoveryEntryMark:
    return DiscoveryEntryMark(
        entry_mark_id=f"em_{mid}", signal_id=f"sig_{mid}", latency_ms=250, size_shares=100,
        entry_target_ts=1_000, entry_actual_ts=1_050 if status == "EXECUTED" else None,
        entry_delay_after_target_ms=50 if status == "EXECUTED" else None,
        entry_best_bid=0.49, entry_best_ask=0.50,
        entry_vwap=0.505 if status == "EXECUTED" else None, entry_slippage=slippage,
        available_ask_liquidity=100.0, entry_fee_total=1.7 if status == "EXECUTED" else None,
        entry_fee_per_share=0.017 if status == "EXECUTED" else None, status=status,
    )


def test_signal_config_summary_denominators():
    marks = [
        _entry("M1", "EXECUTED", slippage=0.005),
        _entry("M1", "EXECUTED", slippage=0.010),
        _entry("M2", "NOT_EXECUTABLE"),
        _entry("M2", "NO_DATA"),
    ]
    summary = compute_signal_config_summary(
        "C1", "BTC", 250, 100.0, marks, market_ids={"M1", "M2"}, control_count=10,
    )
    assert summary.signal_count == 4
    assert summary.market_count == 2
    assert summary.executable_entry_count == 2
    assert summary.not_executable_entry_count == 2
    assert abs(summary.entry_execution_rate - 0.5) < 1e-9  # 2/4, NOT 2/2
    assert abs(summary.mean_entry_delay_ms - 50) < 1e-9
    assert abs(summary.mean_entry_slippage - 0.0075) < 1e-9
    assert summary.control_count == 10


def _response(market_id: str, raw: float | None, status: str = "AVAILABLE") -> SignalResponse:
    fee_adj = (raw - 0.017) if raw is not None else None
    return SignalResponse(
        response_id=f"r_{market_id}_{raw}", entry_mark_id=f"em_{market_id}", signal_id=f"sig_{market_id}",
        signal_config_id="C1", asset="BTC", market_id=market_id, direction="UP",
        latency_ms=250, size_shares=100, horizon_ms=2_000,
        response_target_ts=3_000, response_actual_ts=3_010 if status == "AVAILABLE" else None,
        response_delay_ms=10 if status == "AVAILABLE" else None,
        future_best_bid=0.52, future_best_ask=0.53, future_sell_vwap=(0.505 + raw) if raw is not None else None,
        available_bid_liquidity=100.0, raw_response=raw, fee_adjusted_response=fee_adj,
        response_positive=(raw > 0) if raw is not None else None,
        fee_adjusted_positive=(fee_adj > 0) if fee_adj is not None else None,
        status=status,
    )


def test_signal_response_summary_uplift_and_denominators():
    responses = [
        _response("M1", 0.02), _response("M1", -0.01), _response("M2", 0.03), _response("M2", 0.01),
        _response("M2", None, status="NOT_SELL_EXECUTABLE"),  # must NOT count toward available_count
    ]
    market_id_of = {id(r): r.market_id for r in responses}
    controls = [
        _response("M1", 0.00), _response("M1", -0.005), _response("M2", 0.005), _response("M2", -0.002),
    ]
    control_market_id_of = {id(r): r.market_id for r in controls}

    summary = compute_signal_response_summary(
        "C1", "BTC", 250, 100.0, 2_000, responses, market_id_of, controls, control_market_id_of,
        bootstrap_iterations=200, bootstrap_seed=1337,
    )

    assert summary.available_count == 4  # NOT 5 -- the NOT_SELL_EXECUTABLE one is excluded (contract #14)
    assert summary.unavailable_count == 1
    assert summary.positive_count == 3
    assert summary.negative_count == 1
    assert abs(summary.p_positive - 0.75) < 1e-9
    assert abs(summary.mean_response - (0.02 - 0.01 + 0.03 + 0.01) / 4) < 1e-9
    assert abs(summary.mean_fee_adjusted_response - summary.mean_response + 0.017) < 1e-9

    assert summary.control_available_count == 4
    control_mean = (0.00 - 0.005 + 0.005 - 0.002) / 4
    assert abs(summary.control_mean_response - control_mean) < 1e-9
    assert abs(summary.uplift_mean_response - (summary.mean_response - control_mean)) < 1e-9
    control_p_pos = 1 / 4  # only 0.005 is strictly > 0 among [0.00, -0.005, 0.005, -0.002]
    assert abs(summary.control_p_positive - control_p_pos) < 1e-9
    assert abs(summary.uplift_p_positive - (0.75 - control_p_pos)) < 1e-9

    # bootstrap: fixed seed -> repeated calls give identical CI (contract #33/34).
    summary2 = compute_signal_response_summary(
        "C1", "BTC", 250, 100.0, 2_000, responses, market_id_of, controls, control_market_id_of,
        bootstrap_iterations=200, bootstrap_seed=1337,
    )
    assert summary.bootstrap_ci95_mean_low == summary2.bootstrap_ci95_mean_low
    assert summary.bootstrap_ci95_mean_high == summary2.bootstrap_ci95_mean_high
    assert summary.bootstrap_ci95_mean_low <= summary.mean_response <= summary.bootstrap_ci95_mean_high

    # uplift CI: paired by market, distinct from independently subtracting the two CIs above.
    assert summary.bootstrap_ci95_uplift_low is not None
    assert summary.bootstrap_ci95_uplift_high is not None
    assert summary.bootstrap_ci95_uplift_low <= summary.uplift_mean_response <= summary.bootstrap_ci95_uplift_high
    assert summary.bootstrap_ci95_uplift_low == summary2.bootstrap_ci95_uplift_low


def _path_stats(mid: str, time_to_plus: dict, time_to_minus: dict) -> SignalPathStats:
    return SignalPathStats(
        entry_mark_id=f"em_{mid}", stats_horizon_ms=5_000, mfe=None, mae=None,
        time_to_mfe_ms=None, time_to_mae_ms=None,
        time_to_plus_ms=time_to_plus, time_to_minus_ms=time_to_minus,
    )


def test_hit_summary():
    stats = [
        _path_stats("A", {0.01: 500}, {0.01: None}),
        _path_stats("B", {0.01: None}, {0.01: 900}),
        _path_stats("C", {0.01: None}, {0.01: None}),
    ]
    favorable = compute_hit_summary("C1", "BTC", 250, 100.0, 5_000, 0.01, "FAVORABLE", stats)
    assert favorable.sample_count == 3
    assert favorable.hit_count == 1
    assert abs(favorable.hit_probability - 1 / 3) < 1e-9

    adverse = compute_hit_summary("C1", "BTC", 250, 100.0, 5_000, 0.01, "ADVERSE", stats)
    assert adverse.hit_count == 1


def test_first_passage_summary():
    stats = [
        _path_stats("A", {0.02: 500}, {0.01: None}),   # PLUS_ONLY
        _path_stats("B", {0.02: None}, {0.01: 300}),   # MINUS_ONLY
        _path_stats("C", {0.02: 400}, {0.01: 900}),    # PLUS_FIRST (400 < 900)
        _path_stats("D", {0.02: 900}, {0.01: 400}),    # MINUS_FIRST
        _path_stats("E", {0.02: None}, {0.01: None}),  # NEITHER
        _path_stats("F", {0.02: 500}, {0.01: 500}),    # AMBIGUOUS
    ]
    summary = compute_first_passage_summary("C1", "BTC", 250, 100.0, 5_000, 0.02, 0.01, stats)

    assert summary.sample_count == 6
    assert summary.plus_first_count == 2  # PLUS_ONLY(A) + PLUS_FIRST(C)
    assert summary.minus_first_count == 2  # MINUS_ONLY(B) + MINUS_FIRST(D)
    assert summary.plus_only_count == 1  # A -- distinct from plus_first_count, which folds it in
    assert summary.minus_only_count == 1  # B
    assert summary.ambiguous_count == 1  # F
    assert summary.neither_count == 1  # E
    # denominator excludes NEITHER (contract #17): 2+2+1 = 5
    assert abs(summary.p_plus_first_given_hit - 2 / 5) < 1e-9
    assert abs(summary.p_minus_first_given_hit - 2 / 5) < 1e-9
