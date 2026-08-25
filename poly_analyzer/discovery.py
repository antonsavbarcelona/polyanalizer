"""Offline discovery/replay engine.

Reads raw data (binance_trades, binance_book, polymarket_book,
chainlink_observations) saved by the live collector and replays it, in
causal order (recv_monotonic_ns), through the SAME MarketState update
methods the live collector uses -- byte-identical feature computation, just
with arbitrary window sizes and zero live API dependency. This module never
opens a socket.

    Collector saves facts. Discovery forms hypotheses.

No TP/SL/timeout/PnL concept here either: signal detection is fully
parametrized (DiscoverySignalConfig), and "what happened after" is reported
as executable-return trajectories, MFE/MAE and first-passage times -- never
collapsed into a single trade decision.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field, replace
from typing import Iterator

from .config import SignalConfig
from .execution import vwap_fill
from .features import book_imbalance, trade_flow
from .signal_engine import HysteresisGate
from .state import MarketState
from .timeseries import stdev, time_return

BOOK_LEVELS = 10


# ---------------------------------------------------------------------------
# 1. Signal configuration (fully offline-tunable; see hypothesis discussion)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoverySignalConfig:
    momentum_window_ms: int = 1_000
    volatility_lookback_ms: int = 120_000
    z_threshold: float = 2.5

    flow_window_ms: int = 1_000
    flow_threshold: float = 0.30

    imbalance_depth: int = 5
    imbalance_threshold: float = 0.10

    poly_lag_window_ms: int = 500
    max_poly_reprice: float = 0.01

    reset_z: float = 1.0
    cooldown_ms: int = 3_000

    ask_min: float = 0.20
    ask_max: float = 0.80
    spread_max: float = 0.03
    tte_min_s: float = 90.0
    tte_max_s: float = 780.0

    sigma_min_samples: int = 10


@dataclass
class DiscoveredSignal:
    config_id: str
    market_id: str
    direction: str
    signal_ts: int
    z: float
    momentum: float
    flow: float
    imbalance: float
    poly_ask: float
    poly_bid: float
    poly_spread: float
    remaining_s: float
    ask_change: float


# ---------------------------------------------------------------------------
# 2. Replay: raw rows -> the live MarketState update methods, in causal order
# ---------------------------------------------------------------------------

def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_market_meta(conn: sqlite3.Connection, market_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM markets WHERE market_id=?", (market_id,)).fetchone()
    return dict(row) if row else None


def _load_events(conn: sqlite3.Connection, market_id: str) -> list[tuple[int, str, dict]]:
    """(sort_key, kind, row) ascending. sort_key is recv_monotonic_ns where
    available (true causal receive order within one collector run); falls
    back to recv_ts for older rows that predate that column being backfilled.
    """
    events: list[tuple[int, str, dict]] = []
    tables = [
        ("binance_trades", "trade", "recv_monotonic_ns", "recv_ts"),
        ("binance_book", "book", "recv_monotonic_ns", "recv_ts"),
        ("polymarket_book", "poly", "recv_monotonic_ns", "ts"),
        ("chainlink_observations", "chainlink", "recv_monotonic_ns", "recv_ts"),
    ]
    for table, kind, mono_col, fallback_col in tables:
        for row in conn.execute(f"SELECT * FROM {table} WHERE market_id=?", (market_id,)):
            d = dict(row)
            key = d.get(mono_col)
            if key is None:
                key = d.get(fallback_col, 0)
            events.append((key, kind, d))
    events.sort(key=lambda e: e[0])
    return events


def _apply_event(state: MarketState, kind: str, row: dict) -> int:
    """Mutates state via the exact same on_* methods binance_feed.py /
    polymarket_feed.py / chainlink_feed.py call live, and returns the "now"
    a live on_update() call would have used (always receive time, never the
    source/exchange timestamp -- matches main.py exactly)."""
    if kind == "trade":
        state.on_binance_trade(row["exchange_ts"], row["price"], row["qty"],
                                is_buyer_maker=(row["aggressor_side"] == "SELL"))
        return row["recv_ts"]
    if kind == "book":
        bids = [(row[f"bid_{i}_price"], row[f"bid_{i}_size"]) for i in range(1, BOOK_LEVELS + 1)
                if row[f"bid_{i}_price"] is not None]
        asks = [(row[f"ask_{i}_price"], row[f"ask_{i}_size"]) for i in range(1, BOOK_LEVELS + 1)
                if row[f"ask_{i}_price"] is not None]
        state.on_binance_depth(row["recv_ts"], bids, asks)
        if row["best_bid_price"] is not None and row["best_ask_price"] is not None:
            state.on_binance_book_ticker(row["recv_ts"], row["best_bid_price"], row["best_ask_price"])
        return row["recv_ts"]
    if kind == "poly":
        for direction, prefix in (("UP", "up"), ("DOWN", "down")):
            state.on_poly_book(direction, row["ts"], row[f"{prefix}_best_bid"], row[f"{prefix}_best_ask"],
                                row[f"{prefix}_best_bid_size"], row[f"{prefix}_best_ask_size"])
        return row["ts"]
    if kind == "chainlink":
        state.on_chainlink_twap(row["observation_ts"], row["twap_value"])
        return row["recv_ts"]
    raise ValueError(f"unknown event kind {kind!r}")


def iter_replay(conn: sqlite3.Connection, market_id: str, cfg: DiscoverySignalConfig
                 ) -> Iterator[tuple[int, MarketState, str]]:
    """Yields (now_ms, state, source) after each raw event is applied --
    the offline equivalent of main.py's on_update dispatch. Buffers are
    sized to whatever this specific config needs (never the live 120s/6s/2s
    fixed windows), so arbitrary lookbacks/windows are always valid."""
    max_lookback_ms = max(cfg.volatility_lookback_ms, cfg.momentum_window_ms, cfg.flow_window_ms) + 5_000
    state = MarketState(
        price_history_max_age_ms=max_lookback_ms,
        trade_buffer_max_age_ms=cfg.flow_window_ms + 1_000,
        ask_history_max_age_ms=cfg.poly_lag_window_ms + 1_000,
    )
    meta = load_market_meta(conn, market_id)
    state.market_id = market_id
    if meta:
        state.market_start_ts = meta.get("start_ts")
        state.market_end_ts = meta.get("end_ts")
        state.tick_size = meta.get("tick_size") or 0.01
        state.up_token_id = meta.get("up_token_id")
        state.down_token_id = meta.get("down_token_id")

    for _, kind, row in _load_events(conn, market_id):
        now = _apply_event(state, kind, row)
        yield now, state, kind


# ---------------------------------------------------------------------------
# 3. Feature recomputation at arbitrary windows (generalizes features.py)
# ---------------------------------------------------------------------------

def resampled_returns(price_history, now_ms: int, window_ms: int, lookback_ms: int) -> list[float]:
    """Non-overlapping window_ms-bucketed log-returns over the trailing
    lookback_ms, time-indexed (not tick-indexed) -- the volatility_lookback
    generalization of state.py's fixed 1s-bucket sigma_1s."""
    returns = []
    t = now_ms - lookback_ms
    while t + window_ms <= now_ms:
        p0 = price_history.value_before(t)
        p1 = price_history.value_before(t + window_ms)
        if p0 is not None and p1 is not None and p0[1] > 0:
            returns.append(math.log(p1[1] / p0[1]))
        t += window_ms
    return returns


@dataclass
class DiscoveryFeatures:
    momentum: float | None
    sigma: float | None
    z: float | None
    flow: float | None
    imbalance: float | None


def compute_discovery_features(state: MarketState, now_ms: int, cfg: DiscoverySignalConfig) -> DiscoveryFeatures:
    momentum = time_return(state.price_history, now_ms, cfg.momentum_window_ms)
    returns = resampled_returns(state.price_history, now_ms, cfg.momentum_window_ms, cfg.volatility_lookback_ms)
    sigma = stdev(returns) if len(returns) >= cfg.sigma_min_samples else None
    z = None
    if momentum is not None and sigma is not None and sigma > 0:
        z = momentum / sigma
    return DiscoveryFeatures(
        momentum=momentum, sigma=sigma, z=z,
        flow=trade_flow(state, now_ms, cfg.flow_window_ms),
        imbalance=book_imbalance(state, cfg.imbalance_depth),
    )


# ---------------------------------------------------------------------------
# 4. Signal replay: same decision structure as signal_engine.py, fully
#    parametrized (momentum sign / flow / imbalance / repricing / contract
#    filters / hysteresis), so sweeping configs never requires new live data.
# ---------------------------------------------------------------------------

def _check_confirmations(state: MarketState, feats: DiscoveryFeatures, now_ms: int,
                          direction: str, z: float, cfg: DiscoverySignalConfig) -> DiscoveredSignal | None:
    sign = 1.0 if direction == "UP" else -1.0

    if feats.momentum is None or sign * feats.momentum <= 0:
        return None
    if feats.flow is None or sign * feats.flow < cfg.flow_threshold:
        return None
    if feats.imbalance is None or sign * feats.imbalance < cfg.imbalance_threshold:
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

    past = book.ask_history.value_before(now_ms - cfg.poly_lag_window_ms)
    if past is None:
        return None
    ask_change = book.ask - past[1]
    if ask_change > cfg.max_poly_reprice:
        return None

    return DiscoveredSignal(
        config_id="", market_id=state.market_id, direction=direction, signal_ts=now_ms,
        z=z, momentum=feats.momentum, flow=feats.flow, imbalance=feats.imbalance,
        poly_ask=book.ask, poly_bid=book.bid, poly_spread=book.spread,
        remaining_s=remaining_s, ask_change=ask_change,
    )


def replay_signals(conn: sqlite3.Connection, market_id: str, cfg: DiscoverySignalConfig,
                    config_id: str = "default") -> list[DiscoveredSignal]:
    gate = HysteresisGate(SignalConfig(z_trigger=cfg.z_threshold, z_reset=cfg.reset_z,
                                        cooldown_s=cfg.cooldown_ms / 1000.0))
    found: list[DiscoveredSignal] = []
    for now, state, _kind in iter_replay(conn, market_id, cfg):
        feats = compute_discovery_features(state, now, cfg)
        if not gate.observe(feats.z, now):
            continue
        z = feats.z
        direction = "UP" if z > 0 else "DOWN"
        sig = _check_confirmations(state, feats, now, direction, z, cfg)
        if sig is not None:
            gate.consume(now)
            found.append(replace(sig, config_id=config_id))
    return found


# ---------------------------------------------------------------------------
# 5. Execution marks: realistic entry price for arbitrary simulated latency,
#    using the first Polymarket book state actually available AFTER the
#    target timestamp -- never one from before it (no look-ahead).
# ---------------------------------------------------------------------------

@dataclass
class ExecutionMark:
    latency_ms: int
    target_ts: int
    actual_ts: int | None
    entry_vwap: float | None
    filled_size: float | None
    best_ask: float | None


def extract_levels(row, prefix: str, side: str, levels: int = BOOK_LEVELS) -> list[tuple[float, float]]:
    """row: dict or sqlite3.Row from a `SELECT *` on polymarket_book (both
    support key-based __getitem__; sqlite3.Row has no .get())."""
    out = []
    for i in range(1, levels + 1):
        p, s = row[f"{prefix}_{side}_{i}_price"], row[f"{prefix}_{side}_{i}_size"]
        if p is not None and s is not None:
            out.append((p, s))
    return out


def _find_poly_book_at_or_after(conn: sqlite3.Connection, market_id: str, target_ts: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM polymarket_book WHERE market_id=? AND ts>=? ORDER BY ts ASC LIMIT 1",
        (market_id, target_ts),
    ).fetchone()
    return dict(row) if row else None


def compute_execution_mark(conn: sqlite3.Connection, market_id: str, direction: str,
                            signal_ts: int, latency_ms: int, size: float) -> ExecutionMark:
    target_ts = signal_ts + latency_ms
    row = _find_poly_book_at_or_after(conn, market_id, target_ts)
    prefix = "up" if direction == "UP" else "down"
    if row is None:
        return ExecutionMark(latency_ms, target_ts, None, None, None, None)
    asks = extract_levels(row, prefix, "ask")
    fill = vwap_fill(asks, size)
    return ExecutionMark(
        latency_ms=latency_ms, target_ts=target_ts, actual_ts=row["ts"],
        entry_vwap=fill[0] if fill else None,
        filled_size=fill[1] if fill else None,
        best_ask=row.get(f"{prefix}_best_ask"),
    )


# ---------------------------------------------------------------------------
# 6. Future-response analysis: executable return trajectory, MFE/MAE,
#    first-passage times. No TP/SL -- just what the market actually did.
# ---------------------------------------------------------------------------

DEFAULT_LEVELS_USD = (0.01, 0.02, 0.03, 0.04, 0.05)


@dataclass
class ResponseStats:
    entry_ts: int
    entry_vwap: float
    size: float
    mfe: float | None
    mae: float | None
    time_to_mfe_ms: int | None
    time_to_mae_ms: int | None
    time_to_plus_ms: dict[float, int | None] = field(default_factory=dict)
    time_to_minus_ms: dict[float, int | None] = field(default_factory=dict)
    final_return: float | None = None
    observations: int = 0


def analyze_response(conn: sqlite3.Connection, market_id: str, direction: str,
                      entry_ts: int, entry_vwap: float, size: float,
                      levels_usd: tuple[float, ...] = DEFAULT_LEVELS_USD,
                      horizon_ms: int | None = None) -> ResponseStats:
    """Walks the SAME direction's bid side forward from entry_ts (what we
    could realistically get back selling `size` shares at each point) --
    never best_bid alone, since a spike with insufficient depth must not
    look executable for the full size (RESP-02)."""
    prefix = "up" if direction == "UP" else "down"
    rows = conn.execute(
        "SELECT * FROM polymarket_book WHERE market_id=? AND ts>=? ORDER BY ts ASC",
        (market_id, entry_ts),
    ).fetchall()

    best_ret, best_ts = None, None
    worst_ret, worst_ts = None, None
    time_to_plus = {lvl: None for lvl in levels_usd}
    time_to_minus = {lvl: None for lvl in levels_usd}
    final_return = None
    n = 0

    for row in rows:
        elapsed = row["ts"] - entry_ts
        if horizon_ms is not None and elapsed > horizon_ms:
            break
        bids = extract_levels(row, prefix, "bid")
        fill = vwap_fill(bids, size)
        if fill is None:
            continue
        sell_vwap, _filled = fill
        ret = sell_vwap - entry_vwap
        final_return = ret
        n += 1
        if best_ret is None or ret > best_ret:
            best_ret, best_ts = ret, row["ts"]
        if worst_ret is None or ret < worst_ret:
            worst_ret, worst_ts = ret, row["ts"]
        for lvl in levels_usd:
            if time_to_plus[lvl] is None and ret >= lvl:
                time_to_plus[lvl] = elapsed
            if time_to_minus[lvl] is None and ret <= -lvl:
                time_to_minus[lvl] = elapsed

    return ResponseStats(
        entry_ts=entry_ts, entry_vwap=entry_vwap, size=size,
        mfe=best_ret, mae=worst_ret,
        time_to_mfe_ms=(best_ts - entry_ts) if best_ts is not None else None,
        time_to_mae_ms=(worst_ts - entry_ts) if worst_ts is not None else None,
        time_to_plus_ms=time_to_plus, time_to_minus_ms=time_to_minus,
        final_return=final_return, observations=n,
    )


# ---------------------------------------------------------------------------
# 7. Opposite-side hedge: entry_vwap(direction) + future opposite-ask VWAP
# ---------------------------------------------------------------------------

@dataclass
class HedgePoint:
    ts: int
    elapsed_ms: int
    opposite_ask_vwap: float | None
    combined_cost: float | None  # None if opposite-side liquidity is insufficient


def opposite_side_lock_path(conn: sqlite3.Connection, market_id: str, direction: str,
                             entry_ts: int, entry_vwap: float, size: float,
                             horizon_ms: int | None = None) -> list[HedgePoint]:
    opposite_prefix = "down" if direction == "UP" else "up"
    rows = conn.execute(
        "SELECT * FROM polymarket_book WHERE market_id=? AND ts>=? ORDER BY ts ASC",
        (market_id, entry_ts),
    ).fetchall()
    out = []
    for row in rows:
        elapsed = row["ts"] - entry_ts
        if horizon_ms is not None and elapsed > horizon_ms:
            break
        asks = extract_levels(row, opposite_prefix, "ask")
        fill = vwap_fill(asks, size)
        if fill is None:
            out.append(HedgePoint(row["ts"], elapsed, None, None))
            continue
        opp_vwap = fill[0]
        out.append(HedgePoint(row["ts"], elapsed, opp_vwap, entry_vwap + opp_vwap))
    return out


# ---------------------------------------------------------------------------
# 8. Aggregation across many signals (dependency-free: no numpy/pandas)
# ---------------------------------------------------------------------------

@dataclass
class AggregateStats:
    signal_count: int
    market_count: int
    signals_per_market: float
    p_positive: float | None
    mean_return: float | None
    median_return: float | None
    p25: float | None
    p75: float | None
    p_reach: dict[float, float] = field(default_factory=dict)  # usd level -> P(reached within horizon)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def aggregate_responses(responses: list[ResponseStats], market_ids: set[str],
                         levels_usd: tuple[float, ...] = DEFAULT_LEVELS_USD) -> AggregateStats:
    finals = sorted(r.final_return for r in responses if r.final_return is not None)
    n = len(finals)
    signal_count = len(responses)
    market_count = max(len(market_ids), 1)
    p_reach = {}
    for lvl in levels_usd:
        reached = sum(1 for r in responses if r.time_to_plus_ms.get(lvl) is not None)
        p_reach[lvl] = reached / signal_count if signal_count else 0.0
    return AggregateStats(
        signal_count=signal_count,
        market_count=len(market_ids),
        signals_per_market=signal_count / market_count,
        p_positive=(sum(1 for v in finals if v > 0) / n) if n else None,
        mean_return=(sum(finals) / n) if n else None,
        median_return=_percentile(finals, 0.5) if n else None,
        p25=_percentile(finals, 0.25) if n else None,
        p75=_percentile(finals, 0.75) if n else None,
        p_reach=p_reach,
    )


def list_market_ids(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT market_id FROM markets ORDER BY start_ts")]
