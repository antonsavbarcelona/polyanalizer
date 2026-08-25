from __future__ import annotations

import math
from dataclasses import dataclass

from .config import SignalConfig
from .state import MarketState
from .timeseries import stdev, time_return


@dataclass
class Features:
    momentum_250ms: float | None
    momentum_1s: float | None
    momentum_5s: float | None
    sigma_1s: float | None
    z_1s: float | None
    flow_1s: float | None
    flow_5s: float | None
    book_imbalance: float | None
    vol_5s: float | None
    vol_30s: float | None


def trade_flow(state: MarketState, now_ms: int, window_ms: int) -> float | None:
    cutoff = now_ms - window_ms
    buy_vol = 0.0
    sell_vol = 0.0
    for ts, qty, is_buy in state.trade_buffer:
        if ts < cutoff:
            continue
        if is_buy:
            buy_vol += qty
        else:
            sell_vol += qty
    total = buy_vol + sell_vol
    if total <= 0:
        return None
    return (buy_vol - sell_vol) / total


def book_imbalance(state: MarketState, depth: int) -> float | None:
    if not state.binance_bids or not state.binance_asks:
        return None
    bid_vol = sum(size for _, size in state.binance_bids[:depth])
    ask_vol = sum(size for _, size in state.binance_asks[:depth])
    total = bid_vol + ask_vol
    if total <= 0:
        return None
    return (bid_vol - ask_vol) / total


def book_imbalance_top5(state: MarketState) -> float | None:
    return book_imbalance(state, 5)


def realized_vol(state: MarketState, now_ms: int, window_ms: int, min_samples: int = 3) -> float | None:
    samples = state.price_history.values_in_window(now_ms - window_ms, now_ms)
    if len(samples) < min_samples:
        return None
    returns = []
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        if p0 > 0:
            returns.append(math.log(p1 / p0))
    return stdev(returns)


def sigma_1s(state: MarketState, cfg: SignalConfig) -> float | None:
    n = min(len(state.second_returns), int(cfg.sigma_lookback_s))
    if n < cfg.sigma_min_samples:
        return None
    recent = list(state.second_returns)[-n:]
    return stdev(recent)


def compute_features(state: MarketState, now_ms: int, cfg: SignalConfig) -> Features:
    momentum_250ms = time_return(state.price_history, now_ms, cfg.momentum_window_ms)
    momentum_1s = time_return(state.price_history, now_ms, 1_000)
    momentum_5s = time_return(state.price_history, now_ms, 5_000)

    sig1s = sigma_1s(state, cfg)
    z_1s = None
    if momentum_1s is not None and sig1s is not None and sig1s > 0:
        z_1s = momentum_1s / sig1s

    return Features(
        momentum_250ms=momentum_250ms,
        momentum_1s=momentum_1s,
        momentum_5s=momentum_5s,
        sigma_1s=sig1s,
        z_1s=z_1s,
        flow_1s=trade_flow(state, now_ms, 1_000),
        flow_5s=trade_flow(state, now_ms, 5_000),
        book_imbalance=book_imbalance_top5(state),
        vol_5s=realized_vol(state, now_ms, 5_000),
        vol_30s=realized_vol(state, now_ms, 30_000),
    )


def build_binance_features_row(state: MarketState, now_ms: int, market_id, is_debug: bool) -> dict:
    """Convenience-only normalized snapshot at a fixed default set of
    windows. Explicitly NOT source of truth -- OFFLINE-01/02/03 in
    tests/test_offline_recompute.py prove every one of these fields can be
    reconstructed byte-for-byte from binance_trades/binance_book alone.
    """
    return {
        "ts": now_ms,
        "market_id": market_id,
        "btc_mid": state.btc_mid,
        "return_100ms": time_return(state.price_history, now_ms, 100),
        "return_250ms": time_return(state.price_history, now_ms, 250),
        "return_500ms": time_return(state.price_history, now_ms, 500),
        "return_1s": time_return(state.price_history, now_ms, 1_000),
        "return_2s": time_return(state.price_history, now_ms, 2_000),
        "return_5s": time_return(state.price_history, now_ms, 5_000),
        "vol_10s": realized_vol(state, now_ms, 10_000),
        "vol_30s": realized_vol(state, now_ms, 30_000),
        "vol_60s": realized_vol(state, now_ms, 60_000),
        "vol_120s": realized_vol(state, now_ms, 120_000),
        "flow_250ms": trade_flow(state, now_ms, 250),
        "flow_500ms": trade_flow(state, now_ms, 500),
        "flow_1s": trade_flow(state, now_ms, 1_000),
        "flow_2s": trade_flow(state, now_ms, 2_000),
        "flow_5s": trade_flow(state, now_ms, 5_000),
        "imbalance_top1": book_imbalance(state, 1),
        "imbalance_top5": book_imbalance(state, 5),
        "imbalance_top10": book_imbalance(state, 10),
        "is_debug": 1 if is_debug else 0,
    }
