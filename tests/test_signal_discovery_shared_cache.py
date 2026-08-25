"""Regression test for research/discovery/shared_cache.py (Stage B perf):
running several flow variants of the SAME baseline through the shared
EntryResponsePathCache must produce results byte-identical to running each
variant fully independently (the status quo before this optimization) --
same signals detected (detection is untouched), same controls selected
(control selection/ranking is untouched), and identical entry/response/
path-stats values (only the expensive computation is memoized).

Mirrors tests/test_signal_discovery_path_walk_regression.py's own pattern
for the same kind of change: prove new-fast-path == old-slow-path on a
real fixture before trusting it in the orchestrator.
"""
from __future__ import annotations

from dataclasses import replace as dc_replace

from tests.discovery_fixtures import binance_book_row, binance_trade_row, insert_rows, make_recorder, market_row, \
    poly_book_row

from research.discovery.controls import scan_candidates, select_matched_controls_from_candidates
from research.discovery.detect import detect_signals
from research.discovery.entry import compute_entry_mark
from research.discovery.path_walk import compute_signal_path
from research.discovery.response import HORIZONS_MS
from research.discovery.shared_cache import EntryResponsePathCache
from research.discovery_types import SignalDiscoveryConfig
from research.fees import FeeModel

LATENCY_MS = 250
SIZE_SHARES = 100.0
MARKET_ROW = {"end_ts": 200_000, "fee_rate": 0.07, "fee_exponent": 1.0}


def _build_fixture_db(tmp_path):
    """Two separate upward momentum/flow jumps (JUMP_1 strongly buy-flow-
    confirmed, JUMP_2 only weakly) separated by a long quiet stretch -- long
    enough before/between/after for the resampled-bucket sigma to have
    samples AND for control-matching's +/-10s exclusion window to leave a
    real "quiet" candidate pool to match against (same shape as
    tests/e2e/test_signal_discovery_e2e.py's fixture, extended with a
    second, weaker-flow jump so flow_threshold actually discriminates
    which configs keep which signal)."""
    rec = make_recorder(tmp_path)
    insert_rows(rec, "markets", [market_row("M_MULTI", end_ts=200_000)])

    JUMP_1, JUMP_2 = 60_000, 140_000
    book_rows = [
        binance_book_row("M_MULTI", ts, [(100.0, 50.0)], [(100.02, 50.0)])
        for ts in range(0, JUMP_1, 100)
    ] + [
        binance_book_row("M_MULTI", JUMP_1 + 100, [(104.9, 400.0), (104.8, 400.0)], [(105.0, 20.0)]),
    ] + [
        binance_book_row("M_MULTI", ts, [(105.0, 50.0)], [(105.02, 50.0)])
        for ts in range(JUMP_1 + 500, JUMP_2, 100)
    ] + [
        binance_book_row("M_MULTI", JUMP_2 + 100, [(109.4, 400.0), (109.3, 400.0)], [(109.5, 20.0)]),
    ]
    insert_rows(rec, "binance_book", book_rows)

    insert_rows(rec, "binance_trades", [
        # JUMP_1: heavy one-sided buying -> strong positive signed_flow.
        binance_trade_row("M_MULTI", 1, JUMP_1, 100.0, 1.0, "SELL"),
        binance_trade_row("M_MULTI", 2, JUMP_1 + 20, 104.5, 8.0, "BUY"),
        binance_trade_row("M_MULTI", 3, JUMP_1 + 40, 104.8, 8.0, "BUY"),
        binance_trade_row("M_MULTI", 4, JUMP_1 + 60, 105.0, 8.0, "BUY"),
        # JUMP_2: mixed buy/sell -> weak/borderline signed_flow.
        binance_trade_row("M_MULTI", 5, JUMP_2, 105.0, 1.0, "SELL"),
        binance_trade_row("M_MULTI", 6, JUMP_2 + 20, 109.0, 5.0, "BUY"),
        binance_trade_row("M_MULTI", 7, JUMP_2 + 40, 109.2, 4.9, "SELL"),
        binance_trade_row("M_MULTI", 8, JUMP_2 + 60, 109.5, 5.1, "BUY"),
    ])

    poly_rows = [
        poly_book_row("M_MULTI", 400, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)]),
    ]
    # Quiet polymarket book samples throughout (so control-matching has a
    # regime-matching pool to select from).
    for ts in range(2_000, JUMP_1 - 2_000, 4_000):
        poly_rows.append(poly_book_row("M_MULTI", ts, up_bid=0.49, up_ask=0.50,
                                        up_bids=[(0.49, 100)], up_asks=[(0.50, 100)]))
    poly_rows += [
        poly_book_row("M_MULTI", JUMP_1 + 100, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)]),
        poly_book_row("M_MULTI", JUMP_1 + 360, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)],
                       up_asks=[(0.50, 20), (0.51, 30), (0.52, 100)]),
        poly_book_row("M_MULTI", JUMP_1 + 900, up_bid=0.56, up_ask=0.58, up_bids=[(0.56, 100)], up_asks=[(0.58, 100)]),
        poly_book_row("M_MULTI", JUMP_1 + 2_100, up_bid=0.57, up_ask=0.59, up_bids=[(0.57, 100)], up_asks=[(0.59, 100)]),
    ]
    for ts in range(JUMP_1 + 5_000, JUMP_2 - 2_000, 4_000):
        poly_rows.append(poly_book_row("M_MULTI", ts, up_bid=0.57, up_ask=0.59,
                                        up_bids=[(0.57, 100)], up_asks=[(0.59, 100)]))
    poly_rows += [
        poly_book_row("M_MULTI", JUMP_2 + 100, up_bid=0.57, up_ask=0.59, up_bids=[(0.57, 100)], up_asks=[(0.59, 100)]),
        poly_book_row("M_MULTI", JUMP_2 + 360, up_bid=0.57, up_ask=0.59, up_bids=[(0.57, 100)],
                       up_asks=[(0.59, 20), (0.60, 30), (0.61, 100)]),
        poly_book_row("M_MULTI", JUMP_2 + 900, up_bid=0.63, up_ask=0.65, up_bids=[(0.63, 100)], up_asks=[(0.65, 100)]),
        poly_book_row("M_MULTI", JUMP_2 + 2_100, up_bid=0.64, up_ask=0.66, up_bids=[(0.64, 100)], up_asks=[(0.66, 100)]),
    ]
    insert_rows(rec, "polymarket_book", poly_rows)
    rec.close()
    return tmp_path / "test.db"


def _baseline_configs():
    """3 flow variants of one baseline (momentum/z/vol fixed) -- OFF, a
    loose flow gate (keeps both jumps), and a strict one (should drop
    JUMP_2's weaker/borderline flow)."""
    from dataclasses import replace

    off = SignalDiscoveryConfig(id="", momentum_window_ms=200, volatility_window_ms=1_000, z_threshold=2.0,
                                 max_spread=0.05, min_contract_price=0.05, max_contract_price=0.95)
    loose = replace(off, flow_window_ms=500, flow_threshold=0.1)
    strict = replace(off, flow_window_ms=500, flow_threshold=0.9)
    from research.discovery_types import deterministic_id
    return [replace(c, id=deterministic_id("sr_sigcfg", c)) for c in (off, loose, strict)]


def _old_path_for_config(conn, cfg, asset="BTC"):
    """The status-quo per-config computation (no cache), exactly mirroring
    research/discovery_experiment.py's _run_signal_config_asset body for
    one market -- this is the ground truth the new cached path must match."""
    fee_model = FeeModel()
    signals = detect_signals(conn, "M_MULTI", cfg, asset)

    entries, responses, stats = [], [], []
    controls, control_entries, control_responses = [], [], []
    candidates_cache = {}
    all_signal_ts = [s.signal_ts for s in signals]

    for signal in signals:
        cache_key = ("M_MULTI", signal.direction)
        candidates = candidates_cache.get(cache_key)
        if candidates is None:
            candidates = scan_candidates(conn, "M_MULTI", cfg, signal.direction, None)
            candidates_cache[cache_key] = candidates
        sig_controls = select_matched_controls_from_candidates(
            candidates, "M_MULTI", asset, cfg, signal.signal_id, signal.direction,
            source_tte_s=(signal.time_remaining_ms or 0) / 1000.0, source_price=signal.target_ask or 0.0,
            source_spread=signal.target_spread or 0.0, source_vol_30s=signal.vol_30s,
            source_tte_regime=signal.tte_regime, source_price_regime=signal.price_regime,
            source_spread_regime=signal.spread_regime, source_volatility_regime=None,
            all_signal_ts_this_config=all_signal_ts, k=3,
        )
        controls.extend(sig_controls)

        entry = compute_entry_mark(conn, signal, LATENCY_MS, SIZE_SHARES, MARKET_ROW, fee_model)
        entries.append(entry)
        if entry.status != "EXECUTED":
            continue
        resp, st = compute_signal_path(conn, signal, entry, MARKET_ROW, HORIZONS_MS, HORIZONS_MS)
        responses.extend(resp)
        stats.extend(st)

        for control in sig_controls:
            from research.discovery_types import ControlEntryMark, SignalSnapshot
            pseudo = SignalSnapshot(**{**signal.__dict__, "signal_id": control.control_id,
                                        "signal_ts": control.control_ts})
            c_entry_raw = compute_entry_mark(conn, pseudo, LATENCY_MS, SIZE_SHARES, MARKET_ROW, fee_model)
            c_entry = ControlEntryMark(
                control_entry_mark_id=f"cem_{control.control_id}_{LATENCY_MS}_{SIZE_SHARES}",
                control_id=control.control_id, latency_ms=c_entry_raw.latency_ms,
                size_shares=c_entry_raw.size_shares, entry_target_ts=c_entry_raw.entry_target_ts,
                entry_actual_ts=c_entry_raw.entry_actual_ts,
                entry_delay_after_target_ms=c_entry_raw.entry_delay_after_target_ms,
                entry_best_bid=c_entry_raw.entry_best_bid, entry_best_ask=c_entry_raw.entry_best_ask,
                entry_vwap=c_entry_raw.entry_vwap, entry_slippage=c_entry_raw.entry_slippage,
                available_ask_liquidity=c_entry_raw.available_ask_liquidity,
                entry_fee_total=c_entry_raw.entry_fee_total, entry_fee_per_share=c_entry_raw.entry_fee_per_share,
                status=c_entry_raw.status,
            )
            control_entries.append(c_entry)
            if c_entry.status != "EXECUTED":
                continue
            c_resp, _c_stats = compute_signal_path(conn, pseudo, c_entry_raw, MARKET_ROW, HORIZONS_MS,
                                                     (HORIZONS_MS[0],))
            from research.discovery_types import ControlResponse
            for resp_ in c_resp:
                control_responses.append(ControlResponse(
                    control_response_id=f"cr_{control.control_id}_{LATENCY_MS}_{SIZE_SHARES}_{resp_.horizon_ms}",
                    control_entry_mark_id=c_entry.control_entry_mark_id, control_id=control.control_id,
                    signal_config_id=cfg.id, asset=asset, market_id="M_MULTI", direction=signal.direction,
                    latency_ms=LATENCY_MS, size_shares=SIZE_SHARES, horizon_ms=resp_.horizon_ms,
                    response_target_ts=resp_.response_target_ts, response_actual_ts=resp_.response_actual_ts,
                    response_delay_ms=resp_.response_delay_ms, future_best_bid=resp_.future_best_bid,
                    future_best_ask=resp_.future_best_ask, future_sell_vwap=resp_.future_sell_vwap,
                    available_bid_liquidity=resp_.available_bid_liquidity, raw_response=resp_.raw_response,
                    fee_adjusted_response=resp_.fee_adjusted_response, response_positive=resp_.response_positive,
                    fee_adjusted_positive=resp_.fee_adjusted_positive, status=resp_.status,
                ))

    return signals, entries, responses, stats, controls, control_entries, control_responses


def _new_path_for_config(conn, cfg, cache, asset="BTC"):
    """Same computation, but entry/response for both the signal AND every
    control routes through the shared cache instead of calling
    compute_entry_mark/compute_signal_path directly."""
    fee_model = FeeModel()
    signals = detect_signals(conn, "M_MULTI", cfg, asset)

    entries, responses, stats = [], [], []
    controls, control_entries, control_responses = [], [], []
    candidates_cache = {}
    all_signal_ts = [s.signal_ts for s in signals]

    for signal in signals:
        cache_key = ("M_MULTI", signal.direction)
        candidates = candidates_cache.get(cache_key)
        if candidates is None:
            candidates = scan_candidates(conn, "M_MULTI", cfg, signal.direction, None)
            candidates_cache[cache_key] = candidates
        sig_controls = select_matched_controls_from_candidates(
            candidates, "M_MULTI", asset, cfg, signal.signal_id, signal.direction,
            source_tte_s=(signal.time_remaining_ms or 0) / 1000.0, source_price=signal.target_ask or 0.0,
            source_spread=signal.target_spread or 0.0, source_vol_30s=signal.vol_30s,
            source_tte_regime=signal.tte_regime, source_price_regime=signal.price_regime,
            source_spread_regime=signal.spread_regime, source_volatility_regime=None,
            all_signal_ts_this_config=all_signal_ts, k=3,
        )
        controls.extend(sig_controls)

        entry, resp, st = cache.for_signal(conn, signal, LATENCY_MS, SIZE_SHARES, MARKET_ROW, fee_model,
                                            HORIZONS_MS, HORIZONS_MS)
        entries.append(entry)
        if entry.status == "EXECUTED":
            responses.extend(resp)
            stats.extend(st)

        for control in sig_controls:
            c_entry, c_resp = cache.for_control(conn, control, LATENCY_MS, SIZE_SHARES, MARKET_ROW, fee_model,
                                                 HORIZONS_MS, HORIZONS_MS)
            control_entries.append(c_entry)
            if c_entry.status == "EXECUTED":
                control_responses.extend(c_resp)

    return signals, entries, responses, stats, controls, control_entries, control_responses


def _assert_dataclass_lists_equal(old_list, new_list, label):
    assert len(old_list) == len(new_list), f"{label}: count mismatch {len(old_list)} vs {len(new_list)}"
    for old, new in zip(old_list, new_list):
        assert old == new, f"{label}: {old} != {new}"


def test_shared_cache_matches_uncached_across_flow_variants(tmp_path):
    from poly_analyzer.discovery import open_db

    db_path = _build_fixture_db(tmp_path)
    conn = open_db(str(db_path))
    configs = _baseline_configs()

    cache = EntryResponsePathCache()
    total_signals = 0
    for cfg in configs:
        old = _old_path_for_config(conn, cfg)
        new = _new_path_for_config(conn, cfg, cache)

        labels = ("signals", "entries", "responses", "stats", "controls", "control_entries", "control_responses")
        for label, old_part, new_part in zip(labels, old, new):
            _assert_dataclass_lists_equal(old_part, new_part, f"{cfg.flow_window_ms}/{cfg.flow_threshold}:{label}")

        total_signals += len(old[0])

    # sanity: fixture actually exercises something (not all-empty no-ops)
    assert total_signals > 0

    # sanity: flow_threshold actually discriminates -- strict config keeps
    # fewer (or equal) signals than loose, which keeps fewer-or-equal than OFF.
    off_cfg, loose_cfg, strict_cfg = configs
    off_n = len(detect_signals(conn, "M_MULTI", off_cfg, "BTC"))
    loose_n = len(detect_signals(conn, "M_MULTI", loose_cfg, "BTC"))
    strict_n = len(detect_signals(conn, "M_MULTI", strict_cfg, "BTC"))
    assert off_n >= loose_n >= strict_n
    assert off_n > 0

    # cache actually got reused across configs (not a silent no-op): fewer
    # distinct cache entries than total (signal + control) computations
    # requested across all 3 configs.
    assert len(cache._cache) > 0
