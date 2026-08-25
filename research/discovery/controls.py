"""Deterministic matched controls (IMPLEMENTATION CONTRACT #24-27).

No Python `random`. A control candidate pool is built by re-walking the
SAME market replay used for detection (poly_analyzer.discovery.iter_replay)
and, at every tick, recording the target token's regime labels plus
whether this exact SignalDiscoveryConfig would itself fire there. Matching
is then a deterministic filter (same regimes, not within +/-10s of any real
signal from this config in this market, not itself a signal) + ranked
distance sort, so a repeated run always picks the same K controls.

volatility_regime classification uses vol_30s as the fixed reference metric
(contract #28 doesn't name one; V1 picks vol_30s -- a window already common
across the existing signal-config grids -- and documents it here rather
than leaving it ambiguous)."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from poly_analyzer.discovery import iter_replay
from poly_analyzer.features import realized_vol, trade_flow

from research.discovery.detect import (
    _active_momentum_and_z,
    _active_volume_z,
    _buffer_sizing,
    _check_confirmations,
)
from research.discovery.regimes import classify_price, classify_spread, classify_tte
from research.discovery_types import Control, SignalDiscoveryConfig, VolatilityRegimeBoundaries, deterministic_id

CONTROL_EXCLUSION_WINDOW_MS = 10_000
CONTROLS_PER_SIGNAL = 5

# Normalization divisors for match_distance components (contract #25:
# "normalized abs(...)"). Fixed, documented constants -- not derived from
# the dataset, so match_distance is stable/comparable across runs.
_TTE_NORM_S = 600.0
_PRICE_NORM = 1.0
_SPREAD_NORM = 0.05
_VOL_NORM = 0.01  # ~1% log-return stdev is a reasonable "typical" vol_30s scale


@dataclass
class _Candidate:
    ts: int
    tte_s: float
    price: float
    spread: float
    vol_30s: float | None
    tte_regime: str | None
    price_regime: str | None
    spread_regime: str | None
    volatility_regime: str | None
    eligible: bool  # True iff NOT itself a signal for this config+direction


def scan_candidates(conn: sqlite3.Connection, market_id: str, cfg: SignalDiscoveryConfig, direction: str,
                     vol_boundaries: VolatilityRegimeBoundaries | None) -> list[_Candidate]:
    """The expensive part: a full market re-walk. Its result depends only on
    (market_id, cfg, direction, vol_boundaries) -- NOT on which signal is
    asking -- so callers processing many signals from the same market
    should compute this ONCE per (market_id, direction) and reuse it via
    select_matched_controls_from_candidates(), rather than calling
    select_matched_controls() (which re-scans every time) per signal.
    Measured on real data: with ~100 signals from a single market, this cut
    control-matching cost by roughly the average signals-per-market factor."""
    buffer_cfg = _buffer_sizing(cfg)
    out: list[_Candidate] = []
    for now_ms, state, _kind in iter_replay(conn, market_id, buffer_cfg):
        if state.market_end_ts is None:
            continue
        book = state.token_book(direction)
        if not book.valid or book.ask is None or book.bid is None or book.spread is None:
            continue
        remaining_s = (state.market_end_ts - now_ms) / 1000.0
        vol_30s = realized_vol(state, now_ms, 30_000)

        active_momentum, _sigma, z_score = _active_momentum_and_z(state, now_ms, cfg)
        active_flow = trade_flow(state, now_ms, cfg.flow_window_ms) if cfg.flow_window_ms is not None else None
        volume_z = _active_volume_z(state, now_ms, cfg)
        # _check_confirmations already enforces absolute_move_threshold_bps
        # (if set) unconditionally; it does NOT check z_threshold, since in
        # detect_signals() that's the HysteresisGate's job instead -- so it
        # must be applied here explicitly to decide "would this config fire".
        would_fire = _check_confirmations(state, now_ms, direction, cfg, active_momentum, active_flow, volume_z)
        if would_fire and cfg.z_threshold is not None:
            would_fire = z_score is not None and abs(z_score) >= cfg.z_threshold

        out.append(_Candidate(
            ts=now_ms, tte_s=remaining_s, price=book.ask, spread=book.spread, vol_30s=vol_30s,
            tte_regime=classify_tte(remaining_s), price_regime=classify_price(book.ask),
            spread_regime=classify_spread(book.spread),
            volatility_regime=vol_boundaries.classify(vol_30s) if (vol_boundaries and vol_30s is not None) else None,
            eligible=not would_fire,
        ))
    return out


def select_matched_controls_from_candidates(
    candidates: list[_Candidate],
    market_id: str,
    asset: str,
    cfg: SignalDiscoveryConfig,
    source_signal_id: str,
    direction: str,
    source_tte_s: float,
    source_price: float,
    source_spread: float,
    source_vol_30s: float | None,
    source_tte_regime: str | None,
    source_price_regime: str | None,
    source_spread_regime: str | None,
    source_volatility_regime: str | None,
    all_signal_ts_this_config: list[int],
    k: int = CONTROLS_PER_SIGNAL,
) -> list[Control]:
    """Pure filter/rank over an ALREADY-scanned candidate pool -- no DB
    access, cheap enough to call once per signal even when many signals
    share the same (market_id, direction) scan_candidates() result."""
    pool = []
    for c in candidates:
        if not c.eligible:
            continue
        if any(abs(c.ts - sts) <= CONTROL_EXCLUSION_WINDOW_MS for sts in all_signal_ts_this_config):
            continue
        if c.tte_regime != source_tte_regime:
            continue
        if c.price_regime != source_price_regime:
            continue
        if c.spread_regime != source_spread_regime:
            continue
        if source_volatility_regime is not None and c.volatility_regime != source_volatility_regime:
            continue
        distance = (
            abs(c.tte_s - source_tte_s) / _TTE_NORM_S
            + abs(c.price - source_price) / _PRICE_NORM
            + abs(c.spread - source_spread) / _SPREAD_NORM
            + (abs((c.vol_30s or 0.0) - (source_vol_30s or 0.0)) / _VOL_NORM if source_vol_30s is not None else 0.0)
        )
        pool.append((distance, c))

    pool.sort(key=lambda item: (item[0], item[1].ts))
    selected = pool[:k]

    out: list[Control] = []
    for rank, (distance, c) in enumerate(selected, start=1):
        control_id = deterministic_id(
            "control",
            {"source_signal_id": source_signal_id, "signal_config_id": cfg.id, "control_ts": c.ts},
        )
        out.append(Control(
            control_id=control_id, source_signal_id=source_signal_id, signal_config_id=cfg.id,
            asset=asset, market_id=market_id, direction=direction, control_ts=c.ts,
            match_rank=rank, match_distance=distance,
            tte_regime=c.tte_regime, price_regime=c.price_regime, spread_regime=c.spread_regime,
            volatility_regime=c.volatility_regime,
        ))
    return out


def select_matched_controls(
    conn: sqlite3.Connection,
    market_id: str,
    asset: str,
    cfg: SignalDiscoveryConfig,
    source_signal_id: str,
    direction: str,
    source_tte_s: float,
    source_price: float,
    source_spread: float,
    source_vol_30s: float | None,
    source_tte_regime: str | None,
    source_price_regime: str | None,
    source_spread_regime: str | None,
    source_volatility_regime: str | None,
    all_signal_ts_this_config: list[int],
    vol_boundaries: VolatilityRegimeBoundaries | None,
    k: int = CONTROLS_PER_SIGNAL,
) -> list[Control]:
    """Convenience one-shot form: scans then selects. Callers processing
    many signals from the same market should instead call scan_candidates()
    once and reuse it via select_matched_controls_from_candidates()."""
    candidates = scan_candidates(conn, market_id, cfg, direction, vol_boundaries)
    return select_matched_controls_from_candidates(
        candidates, market_id, asset, cfg, source_signal_id, direction,
        source_tte_s, source_price, source_spread, source_vol_30s,
        source_tte_regime, source_price_regime, source_spread_regime, source_volatility_regime,
        all_signal_ts_this_config, k=k,
    )
