"""Wide-snapshot signal detection (contract #19, #20).

Reuses poly_analyzer.discovery.iter_replay (the SAME MarketState update
methods the live collector and the existing TP/SL discovery pipeline use)
so replay stays byte-identical; everything downstream of iter_replay --
the trigger/confirmation logic and the wide feature snapshot -- is new,
built for this pipeline's much wider SignalDiscoveryConfig (contract #19)
and SignalSnapshot (contract #20) schemas.

Two DELIBERATELY different volatility computations coexist here, by design:
  - active_volatility / z_score: resampled-bucket stdev over
    volatility_window_ms, computed the SAME way as the existing
    poly_analyzer.discovery / research.experiment z-score engine, so a
    signal_config's trigger behavior here is directly comparable to that
    pipeline's.
  - vol_10s..vol_300s (the wide, config-independent snapshot fields):
    poly_analyzer.features.realized_vol -- consecutive-tick stdev within a
    single fixed window. Simpler, and intentionally NOT tied to any one
    signal_config's bucketing choice, since the snapshot must stay
    meaningful regardless of which config triggered the signal.

volume_z is a new metric (not in the old pipeline): a resampled-bucket
z-score of rolling trade volume, analogous to the price z-score. Its
lookback is not a separate config field (contract #19 only lists
volume_window_ms/volume_z_threshold) -- V1 fixes it at
max(60s, VOLUME_Z_MIN_SAMPLES * volume_window_ms), documented here as that
is itself a decision, not left ambiguous.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import replace

from poly_analyzer.discovery import (
    DiscoverySignalConfig as ReplayBufferSizing,
    iter_replay,
)
from poly_analyzer.features import book_imbalance, realized_vol, trade_flow
from poly_analyzer.signal_engine import HysteresisGate
from poly_analyzer.config import SignalConfig as GateConfig
from poly_analyzer.state import MarketState
from poly_analyzer.timeseries import stdev, time_return

from research.discovery.regimes import classify_price, classify_spread, classify_tte
from research.discovery_types import SignalDiscoveryConfig, SignalSnapshot, deterministic_id

# Fixed windows for the wide, config-independent part of the snapshot.
RETURN_WINDOWS_MS = {
    "return_100ms": 100, "return_250ms": 250, "return_500ms": 500, "return_750ms": 750,
    "return_1s": 1_000, "return_1_5s": 1_500, "return_2s": 2_000, "return_3s": 3_000,
    "return_5s": 5_000, "return_10s": 10_000,
}
VOL_WINDOWS_MS = {"vol_10s": 10_000, "vol_30s": 30_000, "vol_60s": 60_000, "vol_120s": 120_000, "vol_300s": 300_000}
FLOW_WINDOWS_MS = {
    "flow_100ms": 100, "flow_250ms": 250, "flow_500ms": 500, "flow_750ms": 750,
    "flow_1s": 1_000, "flow_2s": 2_000, "flow_3s": 3_000, "flow_5s": 5_000, "flow_10s": 10_000,
}
POLY_MOVE_WINDOWS_MS = {
    "poly_move_100ms": 100, "poly_move_250ms": 250, "poly_move_500ms": 500,
    "poly_move_1s": 1_000, "poly_move_2s": 2_000,
}

SNAPSHOT_VOL_LOOKBACK_MS = max(VOL_WINDOWS_MS.values())
SNAPSHOT_FLOW_LOOKBACK_MS = max(FLOW_WINDOWS_MS.values())
SNAPSHOT_POLY_MOVE_LOOKBACK_MS = max(POLY_MOVE_WINDOWS_MS.values())

# Matches research/experiment.py's to_discovery_config() (the TP/SL
# pipeline's own z-score engine, sigma_min_samples=5) rather than
# poly_analyzer.discovery.DiscoverySignalConfig's stricter default of 10,
# so both pipelines' trigger behavior stays comparable at the same window
# sizes.
SIGMA_MIN_SAMPLES = 5
VOLUME_Z_MIN_SAMPLES = 10
VOLUME_Z_MIN_LOOKBACK_MS = 60_000


def _volume_lookback_ms(volume_window_ms: int) -> int:
    return max(VOLUME_Z_MIN_LOOKBACK_MS, VOLUME_Z_MIN_SAMPLES * volume_window_ms)


def _buffer_sizing(cfg: SignalDiscoveryConfig) -> ReplayBufferSizing:
    vol_lookback = max(SNAPSHOT_VOL_LOOKBACK_MS, cfg.volatility_window_ms or 0)
    flow_lookback = max(SNAPSHOT_FLOW_LOOKBACK_MS, cfg.flow_window_ms or 0)
    if cfg.volume_window_ms:
        flow_lookback = max(flow_lookback, _volume_lookback_ms(cfg.volume_window_ms))
    poly_lookback = max(SNAPSHOT_POLY_MOVE_LOOKBACK_MS, cfg.poly_lag_window_ms or 0)
    return ReplayBufferSizing(
        momentum_window_ms=cfg.momentum_window_ms,
        volatility_lookback_ms=vol_lookback,
        flow_window_ms=flow_lookback,
        poly_lag_window_ms=poly_lookback,
    )


def _resampled_returns(price_history, now_ms: int, window_ms: int, lookback_ms: int) -> list[float]:
    returns = []
    t = now_ms - lookback_ms
    while t + window_ms <= now_ms:
        p0 = price_history.value_before(t)
        p1 = price_history.value_before(t + window_ms)
        if p0 is not None and p1 is not None and p0[1] > 0:
            returns.append(math.log(p1[1] / p0[1]))
        t += window_ms
    return returns


def _resampled_volume_sums(trade_buffer, now_ms: int, window_ms: int, lookback_ms: int) -> list[float]:
    rows = list(trade_buffer)
    sums = []
    t = now_ms - lookback_ms
    while t + window_ms <= now_ms:
        t1 = t + window_ms
        sums.append(sum(qty for ts, qty, _ in rows if t <= ts < t1))
        t = t1
    return sums


def _current_volume(trade_buffer, now_ms: int, window_ms: int) -> float:
    cutoff = now_ms - window_ms
    return sum(qty for ts, qty, _ in trade_buffer if ts >= cutoff)


def _microprice_bias(state: MarketState) -> float | None:
    if not state.binance_bids or not state.binance_asks:
        return None
    bid_p, bid_s = state.binance_bids[0]
    ask_p, ask_s = state.binance_asks[0]
    total = bid_s + ask_s
    if total <= 0:
        return None
    microprice = (bid_p * ask_s + ask_p * bid_s) / total
    mid = (bid_p + ask_p) / 2.0
    return microprice - mid


def _distance_to_reference_bps(mid: float | None, reference: float | None) -> float | None:
    if mid is None or reference is None or reference == 0:
        return None
    return (mid - reference) / reference * 10_000.0


def _active_momentum_and_z(state: MarketState, now_ms: int, cfg: SignalDiscoveryConfig
                            ) -> tuple[float | None, float | None, float | None]:
    """(active_momentum_value, active_volatility, z_score) -- resampled-bucket
    method, matching the existing poly_analyzer.discovery z-score engine."""
    momentum = time_return(state.price_history, now_ms, cfg.momentum_window_ms)
    if cfg.volatility_window_ms is None:
        return momentum, None, None
    returns = _resampled_returns(state.price_history, now_ms, cfg.momentum_window_ms, cfg.volatility_window_ms)
    sigma = stdev(returns) if len(returns) >= SIGMA_MIN_SAMPLES else None
    z = momentum / sigma if (momentum is not None and sigma is not None and sigma > 0) else None
    return momentum, sigma, z


def _active_volume_z(state: MarketState, now_ms: int, cfg: SignalDiscoveryConfig) -> float | None:
    if cfg.volume_window_ms is None:
        return None
    lookback = _volume_lookback_ms(cfg.volume_window_ms)
    sums = _resampled_volume_sums(state.trade_buffer, now_ms, cfg.volume_window_ms, lookback)
    if len(sums) < VOLUME_Z_MIN_SAMPLES:
        return None
    sigma = stdev(sums)
    if sigma is None or sigma <= 0:
        return None
    current = _current_volume(state.trade_buffer, now_ms, cfg.volume_window_ms)
    mean = sum(sums) / len(sums)
    return (current - mean) / sigma


def build_snapshot(state: MarketState, now_ms: int, cfg: SignalDiscoveryConfig, direction: str,
                    asset: str, active_momentum: float | None, active_volatility: float | None,
                    z_score: float | None, active_flow: float | None, volume_z: float | None,
                    signal_config_id: str) -> SignalSnapshot:
    book = state.token_book(direction)
    remaining_s = (state.market_end_ts - now_ms) / 1000.0 if state.market_end_ts is not None else None
    elapsed_ms = (now_ms - state.market_start_ts) if state.market_start_ts is not None else None
    remaining_ms = int(remaining_s * 1000) if remaining_s is not None else None

    returns = {name: time_return(state.price_history, now_ms, w) for name, w in RETURN_WINDOWS_MS.items()}
    vols = {name: realized_vol(state, now_ms, w) for name, w in VOL_WINDOWS_MS.items()}
    flows = {name: trade_flow(state, now_ms, w) for name, w in FLOW_WINDOWS_MS.items()}
    poly_moves = {name: time_return(book.ask_history, now_ms, w) for name, w in POLY_MOVE_WINDOWS_MS.items()}

    signal_id = deterministic_id(
        "signal",
        {"signal_config_id": signal_config_id, "asset": asset, "market_id": state.market_id,
         "signal_ts": now_ms, "direction": direction},
    )

    return SignalSnapshot(
        signal_id=signal_id,
        signal_config_id=signal_config_id,
        asset=asset,
        market_id=state.market_id,
        direction=direction,
        signal_ts=now_ms,
        binance_mid=state.btc_mid,
        active_momentum_value=active_momentum,
        active_volatility=active_volatility,
        z_score=z_score,
        active_flow=active_flow,
        volume_z=volume_z,
        imbalance_top1=book_imbalance(state, 1),
        imbalance_top3=book_imbalance(state, 3),
        imbalance_top5=book_imbalance(state, 5),
        imbalance_top10=book_imbalance(state, 10),
        microprice_bias=_microprice_bias(state),
        target_bid=book.bid,
        target_ask=book.ask,
        target_spread=book.spread,
        target_bid_size=book.bid_size,
        target_ask_size=book.ask_size,
        time_remaining_ms=remaining_ms,
        time_elapsed_ms=elapsed_ms,
        reference_price=state.chainlink_reference_price,
        chainlink_twap=state.chainlink_twap,
        distance_to_reference_bps=_distance_to_reference_bps(state.btc_mid, state.chainlink_reference_price),
        volatility_regime=None,  # filled in by the caller (needs per-asset boundaries; contract #28)
        tte_regime=classify_tte(remaining_s),
        price_regime=classify_price(book.ask),
        spread_regime=classify_spread(book.spread),
        **returns, **vols, **flows, **poly_moves,
    )


def _check_confirmations(state: MarketState, now_ms: int, direction: str, cfg: SignalDiscoveryConfig,
                          active_momentum: float | None, active_flow: float | None,
                          volume_z: float | None) -> bool:
    sign = 1.0 if direction == "UP" else -1.0

    if active_momentum is None or sign * active_momentum <= 0:
        return False
    if cfg.absolute_move_threshold_bps is not None:
        if active_momentum is None or abs(active_momentum) * 10_000.0 < cfg.absolute_move_threshold_bps:
            return False
    if cfg.flow_threshold is not None:
        if active_flow is None or sign * active_flow < cfg.flow_threshold:
            return False
    if cfg.volume_z_threshold is not None:
        if volume_z is None or volume_z < cfg.volume_z_threshold:
            return False
    if cfg.imbalance_threshold is not None:
        imbalance = book_imbalance(state, cfg.imbalance_depth)
        if imbalance is None or sign * imbalance < cfg.imbalance_threshold:
            return False

    if state.market_end_ts is None:
        return False
    remaining_s = (state.market_end_ts - now_ms) / 1000.0
    remaining_ms = remaining_s * 1000.0
    if cfg.min_time_remaining_ms is not None and remaining_ms < cfg.min_time_remaining_ms:
        return False
    if cfg.max_time_remaining_ms is not None and remaining_ms > cfg.max_time_remaining_ms:
        return False

    book = state.token_book(direction)
    if not book.valid or book.ask is None or book.bid is None:
        return False
    if cfg.min_contract_price is not None and book.ask < cfg.min_contract_price:
        return False
    if cfg.max_contract_price is not None and book.ask > cfg.max_contract_price:
        return False
    if cfg.max_spread is not None and (book.spread is None or book.spread > cfg.max_spread):
        return False

    if cfg.poly_lag_window_ms is not None and cfg.max_poly_reprice is not None:
        past = book.ask_history.value_before(now_ms - cfg.poly_lag_window_ms)
        if past is None:
            return False
        if (book.ask - past[1]) > cfg.max_poly_reprice:
            return False

    return True


def detect_signals(conn: sqlite3.Connection, market_id: str, cfg: SignalDiscoveryConfig,
                    asset: str) -> list[SignalSnapshot]:
    """Replays one market through the same MarketState update methods the
    live collector uses, and returns every SignalSnapshot where `cfg`'s
    trigger + confirmations fired. Mirrors poly_analyzer.discovery's
    hysteresis-gate structure so anti-spam behavior is consistent across
    both discovery pipelines."""
    gate_trigger = cfg.z_threshold if cfg.z_threshold is not None else cfg.absolute_move_threshold_bps
    gate = HysteresisGate(GateConfig(z_trigger=gate_trigger, z_reset=cfg.reset_threshold,
                                      cooldown_s=cfg.cooldown_ms / 1000.0))
    buffer_cfg = _buffer_sizing(cfg)
    found: list[SignalSnapshot] = []

    for now_ms, state, _kind in iter_replay(conn, market_id, buffer_cfg):
        active_momentum, active_volatility, z_score = _active_momentum_and_z(state, now_ms, cfg)
        volume_z = _active_volume_z(state, now_ms, cfg)
        gate_metric = z_score if cfg.z_threshold is not None else (
            active_momentum * 10_000.0 if active_momentum is not None else None
        )
        if not gate.observe(gate_metric, now_ms):
            continue
        direction = "UP" if gate_metric > 0 else "DOWN"
        active_flow = trade_flow(state, now_ms, cfg.flow_window_ms) if cfg.flow_window_ms is not None else None
        if not _check_confirmations(state, now_ms, direction, cfg, active_momentum, active_flow, volume_z):
            continue
        gate.consume(now_ms)
        snapshot = build_snapshot(state, now_ms, cfg, direction, asset, active_momentum, active_volatility,
                                   z_score, active_flow, volume_z, cfg.id)
        found.append(snapshot)

    return found
