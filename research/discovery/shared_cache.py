"""Cross-flow-variant memoization of entry-mark + future-path computation
(Stage B perf: task flagged 1.5-2.5 days as "still довольно тяжёлый" and
asked for profiling before Stage C/D/E).

Profiling one real Stage B unit showed ~91% of its wall time in
compute_entry_mark/compute_signal_path calls (research/discovery/entry.py,
research/discovery/path_walk.py) -- run once per SIGNAL and once per every
one of its matched CONTROLS. Both functions are proven pure functions of
(market_id, direction, ts, latency_ms, size_shares, market_row, fee_model)
-- neither reads signal_config_id or any cfg field; signal_config_id only
gets stamped onto the OUTPUT rows for labeling (verified by reading both
call sites end-to-end, not assumed).

Stage B evaluates 49 flow variants (1 OFF + 8 windows x 6 thresholds) per
momentum/z/vol baseline. Those variants overwhelmingly share (market_id,
direction, ts) instants: as SIGNALS (looser variants' signal sets heavily
overlap tighter ones' -- though NOT a strict subset relationship, since the
per-config HysteresisGate's armed/cooldown state depends on which earlier
candidates each variant's OWN confirmations accepted -- so detection is
still run independently and honestly per config, never skipped) and even
more so as matched CONTROLS (controls.py's eligibility/exclusion logic is
genuinely config-specific -- deliberately NOT touched here -- but the
underlying quiet-period candidate pool it selects from is mostly shared
across variants of one baseline, so the same control_ts recurs constantly).

This cache makes a second hit at the same (market_id, direction, ts,
latency_ms, size_shares) key free -- no second DB query, no second future-
path walk -- while leaving detection AND control selection/ranking
byte-identical to running each of the 49 configs fully independently.
Regression-tested for exactly that in
tests/test_signal_discovery_shared_cache.py.

Scope: one instance per worker job (one baseline x asset), never persisted
or shared across processes -- a plain in-memory dict.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from research.discovery.entry import compute_entry_mark
from research.discovery.path_walk import compute_signal_path
from research.discovery_types import (
    Control,
    ControlEntryMark,
    ControlResponse,
    DiscoveryEntryMark,
    SignalPathStats,
    SignalResponse,
    SignalSnapshot,
    deterministic_id,
)
from research.fees import FeeModel

# A throwaway SignalSnapshot used only to drive compute_entry_mark/
# compute_signal_path's first (cache-miss) computation -- every field but
# market_id/direction/signal_ts/signal_id is unread by those two functions
# (verified by inspection), so the rest can be None/placeholder.
_UNUSED_SNAPSHOT_FIELDS: dict[str, Any] = dict(
    binance_mid=None, return_100ms=None, return_250ms=None, return_500ms=None, return_750ms=None,
    return_1s=None, return_1_5s=None, return_2s=None, return_3s=None, return_5s=None, return_10s=None,
    active_momentum_value=None, vol_10s=None, vol_30s=None, vol_60s=None, vol_120s=None, vol_300s=None,
    active_volatility=None, z_score=None, flow_100ms=None, flow_250ms=None, flow_500ms=None, flow_750ms=None,
    flow_1s=None, flow_2s=None, flow_3s=None, flow_5s=None, flow_10s=None, active_flow=None, volume_z=None,
    imbalance_top1=None, imbalance_top3=None, imbalance_top5=None, imbalance_top10=None, microprice_bias=None,
    target_bid=None, target_ask=None, target_spread=None, target_bid_size=None, target_ask_size=None,
    poly_move_100ms=None, poly_move_250ms=None, poly_move_500ms=None, poly_move_1s=None, poly_move_2s=None,
    time_remaining_ms=None, time_elapsed_ms=None, reference_price=None, chainlink_twap=None,
    distance_to_reference_bps=None, volatility_regime=None, tte_regime="0-30s", price_regime=None,
    spread_regime=None,
)


class EntryResponsePathCache:
    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[DiscoveryEntryMark, list[SignalResponse], list[SignalPathStats]]] = {}
        self.hits = 0
        self.misses = 0

    def clear_market(self) -> None:
        """Every cache key is (market_id, direction, ts, latency_ms,
        size_shares) -- once a caller has finished every config's pass over
        one market_id, no later market will ever share that market_id
        again, so those entries can never be hit again. MUST be called
        after each market in _run_baseline_asset's outer loop -- an
        unbounded cache here caused a real overnight MemoryError that
        crashed a Stage B run (BrokenProcessPool cascade, ~1000 units
        wrongly marked FAILED). Clearing lets memory stay bounded by "one
        market's worth of signal/control timestamps x 8 configs in a
        chunk", not "every market this job has ever touched"."""
        self._cache.clear()

    def _raw(
        self, conn, market_id: str, direction: str, ts: int, latency_ms: int, size_shares: float,
        market_row: dict[str, Any], fee_model: FeeModel, response_horizons: tuple[int, ...],
        stats_horizons: tuple[int, ...],
    ) -> tuple[DiscoveryEntryMark, list[SignalResponse], list[SignalPathStats]]:
        key = (market_id, direction, ts, latency_ms, size_shares)
        hit = self._cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1

        placeholder = SignalSnapshot(
            signal_id=f"cache_{market_id}_{direction}_{ts}_{latency_ms}_{size_shares}",
            signal_config_id="", asset="", market_id=market_id, direction=direction, signal_ts=ts,
            **_UNUSED_SNAPSHOT_FIELDS,
        )
        entry = compute_entry_mark(conn, placeholder, latency_ms, size_shares, market_row, fee_model)
        responses: list[SignalResponse] = []
        stats: list[SignalPathStats] = []
        if entry.status == "EXECUTED":
            responses, stats = compute_signal_path(
                conn, placeholder, entry, market_row, response_horizons, stats_horizons,
            )
        result = (entry, responses, stats)
        self._cache[key] = result
        return result

    def for_signal(
        self, conn, signal: SignalSnapshot, latency_ms: int, size_shares: float, market_row: dict[str, Any],
        fee_model: FeeModel, response_horizons: tuple[int, ...], stats_horizons: tuple[int, ...],
    ) -> tuple[DiscoveryEntryMark, list[SignalResponse], list[SignalPathStats]]:
        raw_entry, raw_responses, raw_stats = self._raw(
            conn, signal.market_id, signal.direction, signal.signal_ts, latency_ms, size_shares, market_row,
            fee_model, response_horizons, stats_horizons,
        )
        entry_mark_id = deterministic_id(
            "entry_mark", {"signal_id": signal.signal_id, "latency_ms": latency_ms, "size_shares": size_shares},
        )
        entry = replace(raw_entry, entry_mark_id=entry_mark_id, signal_id=signal.signal_id)
        responses = [
            replace(
                r, response_id=deterministic_id("signal_response", {"entry_mark_id": entry_mark_id,
                                                                      "horizon_ms": r.horizon_ms}),
                entry_mark_id=entry_mark_id, signal_id=signal.signal_id, signal_config_id=signal.signal_config_id,
                asset=signal.asset, market_id=signal.market_id, direction=signal.direction,
            )
            for r in raw_responses
        ]
        stats = [replace(s, entry_mark_id=entry_mark_id) for s in raw_stats]
        return entry, responses, stats

    def for_control(
        self, conn, control: Control, latency_ms: int, size_shares: float, market_row: dict[str, Any],
        fee_model: FeeModel, response_horizons: tuple[int, ...], stats_horizons: tuple[int, ...],
    ) -> tuple[ControlEntryMark, list[ControlResponse]]:
        raw_entry, raw_responses, _raw_stats = self._raw(
            conn, control.market_id, control.direction, control.control_ts, latency_ms, size_shares, market_row,
            fee_model, response_horizons, stats_horizons,
        )
        cem_id = f"cem_{control.control_id}_{latency_ms}_{size_shares}"
        c_entry = ControlEntryMark(
            control_entry_mark_id=cem_id, control_id=control.control_id, latency_ms=latency_ms,
            size_shares=size_shares, entry_target_ts=raw_entry.entry_target_ts,
            entry_actual_ts=raw_entry.entry_actual_ts,
            entry_delay_after_target_ms=raw_entry.entry_delay_after_target_ms,
            entry_best_bid=raw_entry.entry_best_bid, entry_best_ask=raw_entry.entry_best_ask,
            entry_vwap=raw_entry.entry_vwap, entry_slippage=raw_entry.entry_slippage,
            available_ask_liquidity=raw_entry.available_ask_liquidity,
            entry_fee_total=raw_entry.entry_fee_total, entry_fee_per_share=raw_entry.entry_fee_per_share,
            status=raw_entry.status,
        )
        c_responses = [
            ControlResponse(
                control_response_id=f"cr_{control.control_id}_{latency_ms}_{size_shares}_{r.horizon_ms}",
                control_entry_mark_id=cem_id, control_id=control.control_id,
                signal_config_id=control.signal_config_id, asset=control.asset, market_id=control.market_id,
                direction=control.direction, latency_ms=latency_ms, size_shares=size_shares,
                horizon_ms=r.horizon_ms, response_target_ts=r.response_target_ts,
                response_actual_ts=r.response_actual_ts, response_delay_ms=r.response_delay_ms,
                future_best_bid=r.future_best_bid, future_best_ask=r.future_best_ask,
                future_sell_vwap=r.future_sell_vwap, available_bid_liquidity=r.available_bid_liquidity,
                raw_response=r.raw_response, fee_adjusted_response=r.fee_adjusted_response,
                response_positive=r.response_positive, fee_adjusted_positive=r.fee_adjusted_positive,
                status=r.status,
            )
            for r in raw_responses
        ]
        return c_entry, c_responses
