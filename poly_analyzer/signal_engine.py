"""H1 signal engine: deterministic, no ML, exactly as specified in hypo.md.

    UP:   z_1s >= +2.5   AND momentum_250ms > 0  AND flow_1s >= +0.25
          AND imbalance >= +0.10 AND target ask hasn't repriced up >1c/500ms
    DOWN: mirror image with all signs flipped.

Plus contract filters (ask in [.20,.80], spread<=.03, TTE in [90,780]s).

Anti-spam is a single hysteresis gate on |z_1s| (trigger=2.5, reset=1.0,
cooldown=3s) shared by both directions, edge-triggered on entering an
excursion above the trigger threshold -- see HysteresisGate for the exact
state machine and why a failed-confirmation excursion still consumes the
edge (it must not re-evaluate every tick while z stays elevated).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import SignalConfig
from .features import Features
from .state import MarketState


@dataclass
class Signal:
    direction: str  # "UP" | "DOWN"
    ts_ms: int
    token_id: str
    z_1s: float
    momentum_250ms: float
    flow_1s: float
    imbalance: float
    poly_ask: float
    poly_bid: float
    poly_spread: float
    remaining_s: float
    ask_change_500ms: float


class HysteresisGate:
    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self.armed = True
        self.in_excursion = False
        self.last_signal_ts_ms: int | None = None

    def observe(self, z_1s: float | None, now_ms: int) -> bool:
        cfg = self.cfg
        if z_1s is None:
            abs_z = None
        else:
            abs_z = abs(z_1s)

        is_trigger_edge = False
        if abs_z is not None:
            if not self.in_excursion and abs_z >= cfg.z_trigger:
                self.in_excursion = True
                if self.armed:
                    is_trigger_edge = True
            elif self.in_excursion and abs_z < cfg.z_trigger:
                self.in_excursion = False

            if not self.armed and abs_z < cfg.z_reset:
                cooldown_elapsed = (
                    self.last_signal_ts_ms is None
                    or (now_ms - self.last_signal_ts_ms) >= cfg.cooldown_s * 1000
                )
                if cooldown_elapsed:
                    self.armed = True

        return is_trigger_edge

    def consume(self, now_ms: int) -> None:
        self.armed = False
        self.last_signal_ts_ms = now_ms


class SignalEngine:
    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self.gate = HysteresisGate(cfg)

    def evaluate(self, state: MarketState, features: Features, now_ms: int) -> Signal | None:
        is_trigger_edge = self.gate.observe(features.z_1s, now_ms)
        if not is_trigger_edge:
            return None

        z = features.z_1s
        direction = "UP" if z > 0 else "DOWN"
        signal = self._check_confirmations(state, features, now_ms, direction, z)
        if signal is not None:
            self.gate.consume(now_ms)
        return signal

    def _check_confirmations(self, state: MarketState, features: Features, now_ms: int,
                              direction: str, z: float) -> Signal | None:
        cfg = self.cfg
        sign = 1.0 if direction == "UP" else -1.0

        if features.momentum_250ms is None or sign * features.momentum_250ms <= 0:
            return None
        if features.flow_1s is None or sign * features.flow_1s < cfg.flow_threshold:
            return None
        if features.book_imbalance is None or sign * features.book_imbalance < cfg.imbalance_threshold:
            return None

        if state.market_end_ts is None:
            return None
        remaining_s = (state.market_end_ts - now_ms) / 1000.0
        if not (cfg.tte_min_s <= remaining_s <= cfg.tte_max_s):
            return None

        book = state.token_book(direction)
        token_id = state.up_token_id if direction == "UP" else state.down_token_id
        if token_id is None or not book.valid or book.ask is None or book.bid is None:
            return None
        if not (cfg.ask_min <= book.ask <= cfg.ask_max):
            return None
        if book.spread is None or book.spread > cfg.spread_max:
            return None

        past = book.ask_history.value_before(now_ms - cfg.repricing_window_ms)
        if past is None:
            return None  # can't confirm "not yet repriced" -> don't trade blind
        ask_change = book.ask - past[1]
        if ask_change > cfg.repricing_max_move:
            return None

        return Signal(
            direction=direction,
            ts_ms=now_ms,
            token_id=token_id,
            z_1s=z,
            momentum_250ms=features.momentum_250ms,
            flow_1s=features.flow_1s,
            imbalance=features.book_imbalance,
            poly_ask=book.ask,
            poly_bid=book.bid,
            poly_spread=book.spread,
            remaining_s=remaining_s,
            ask_change_500ms=ask_change,
        )
