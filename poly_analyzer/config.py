"""Discovery-stage config for H1 (does Binance lead Polymarket repricing?).

Deliberately holds NO trading parameters (no TP/SL/timeout/fees) -- per the
2026-08-22 redesign, the collector only detects signals and records the
full subsequent market path + official settlement. Every question about
entries, exits, thresholds or holding period is answered later by querying
the recorded data, never decided live.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Loads .env (if present) into the process environment BEFORE anything below
# reads os.environ -- must happen at import time, not inside build_config(),
# since the module-level CONFIG singleton further down calls build_config()
# eagerly on import. Safe to call even with no .env file (no-op) or when
# real env vars are already set by the deploy platform (Railway etc. --
# load_dotenv() never overrides an already-set variable by default).
load_dotenv()


@dataclass(frozen=True)
class SignalConfig:
    z_trigger: float = 2.5
    z_reset: float = 1.0
    cooldown_s: float = 3.0

    flow_threshold: float = 0.25
    imbalance_threshold: float = 0.10

    momentum_window_ms: int = 250
    repricing_window_ms: int = 500
    repricing_max_move: float = 0.01

    ask_min: float = 0.20
    ask_max: float = 0.80
    spread_max: float = 0.03

    tte_min_s: float = 90.0
    tte_max_s: float = 780.0

    sigma_lookback_s: float = 120.0
    sigma_min_samples: int = 30


@dataclass(frozen=True)
class RecordingConfig:
    # Executable-depth VWAP tiers used only for the console's live display
    # marks (see display_latencies_ms below) -- raw storage keeps full
    # top-10 depth, so offline analysis is not limited to these sizes.
    vwap_size_tiers: tuple[float, ...] = (25.0, 100.0, 500.0)
    tick_size: float = 0.01

    # binance_book/binance_features cadence: Binance's own depth10@100ms
    # push already paces raw book writes: this only governs the derived
    # binance_features convenience snapshot.
    binance_features_interval_ms: int = 100

    # polymarket_book gets a row on every actual top-of-book change, plus a
    # heartbeat at this cadence so quiet periods still leave a trail.
    poly_heartbeat_ms: int = 250

    # Console-only "what would we have seen at signal_ts + Xms" marks.
    # Not stored as separate rows -- any latency is a query parameter over
    # the raw tables after the fact.
    display_latencies_ms: tuple[int, ...] = (100, 250, 500)


@dataclass(frozen=True)
class MarketConfig:
    asset_slug_prefix: str = "btc-updown-15m"
    window_s: int = 900
    chainlink_symbol: str = "btc/usd"
    chainlink_twap_window_s: int = 60  # 15m/4h markets settle on the 60s TWAP
    binance_symbol: str = "btcusdt"

    gamma_api_base: str = "https://gamma-api.polymarket.com"
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    rtds_ws_url: str = "wss://ws-live-data.polymarket.com"
    binance_ws_url: str = (
        "wss://stream.binance.com:9443/stream?streams={symbol}@aggTrade/"
        "{symbol}@bookTicker/{symbol}@depth10@100ms"
    )

    # Official on-chain resolution lags market end (observed ~90s live) --
    # wait before the first poll, then retry until resolved or timeout.
    settlement_start_delay_s: float = 20.0
    settlement_poll_interval_s: float = 10.0
    settlement_poll_timeout_s: float = 300.0


@dataclass(frozen=True)
class RecorderConfig:
    db_path: str = "data/analyzer.db"
    write_batch_size: int = 100
    write_interval_s: float = 1.0

    # If set, Recorder writes to this Postgres database (via asyncpg)
    # instead of the SQLite file at db_path -- lets btc/eth/sol collectors
    # share ONE durable, query-while-running database for a long
    # unattended collection run instead of three separate SQLite files.
    # db_path is still used by tests/offline tooling that construct a
    # Recorder directly without going through build_config().
    dsn: str | None = None
    # Stamped onto every row when writing to Postgres (multiple assets can
    # safely share one Postgres database/schema, unlike the SQLite path
    # where isolation came from one file per asset).
    asset: str = "BTC"


@dataclass(frozen=True)
class Config:
    signal: SignalConfig = field(default_factory=SignalConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    debug_mode: bool = False


@dataclass(frozen=True)
class AssetSpec:
    binance_symbol: str
    chainlink_symbol: str
    slug_prefix: str


# Same market mechanics for every asset -- confirmed live: identical slug
# pattern "{asset}-updown-15m-{window_start_unix}", same 60s Chainlink TWAP
# resolution, same tick size. Only the symbols change.
ASSETS: dict[str, AssetSpec] = {
    "btc": AssetSpec("btcusdt", "btc/usd", "btc-updown-15m"),
    "eth": AssetSpec("ethusdt", "eth/usd", "eth-updown-15m"),
    "sol": AssetSpec("solusdt", "sol/usd", "sol-updown-15m"),
}


def build_config(asset: str, debug_mode: bool = False) -> Config:
    """One Config per asset, including its own db file -- running btc/eth/sol
    collectors in parallel (separate processes) must never share a database
    or a WebSocket subscription."""
    key = asset.lower()
    if key not in ASSETS:
        raise ValueError(f"unknown asset {asset!r}, expected one of {sorted(ASSETS)}")
    spec = ASSETS[key]
    market = MarketConfig(
        asset_slug_prefix=spec.slug_prefix,
        chainlink_symbol=spec.chainlink_symbol,
        binance_symbol=spec.binance_symbol,
    )
    recorder = RecorderConfig(
        db_path=f"data/analyzer_{key}.db", dsn=os.environ.get("POLY_ANALYZER_DSN"), asset=key.upper(),
    )
    return Config(market=market, recorder=recorder, debug_mode=debug_mode)


CONFIG = build_config("btc")
