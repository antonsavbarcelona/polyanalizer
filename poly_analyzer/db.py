"""Recorder: SQLite (default, used by tests and offline tooling) or
Postgres (when RecorderConfig.dsn is set -- see config.build_config()).
Connectors/engines never touch the DB directly -- they enqueue rows on an
asyncio.Queue and a single writer task batches them, so a slow disk/network
never blocks the WebSocket read loops.

2026-08-22 raw-data redesign: the collector's job is to save facts, not
features. Binance trades/book, Polymarket book and Chainlink observations
are stored close to raw (top-10 depth, per-event) so any feature definition
(momentum window, volatility lookback, flow window, imbalance depth...) can
be recomputed offline in discovery.py without a new live collection run.
`binance_features` is an explicitly-optional, explicitly-recomputable
convenience snapshot -- never treated as source of truth.

2026-08-25: added a Postgres backend so btc/eth/sol collectors can share
ONE durable, query-while-running database for a multi-day unattended
collection run instead of three separate SQLite files. Selected at runtime
via RecorderConfig.dsn (env var POLY_ANALYZER_DSN) -- SQLite stays the
default so existing tests/offline tooling that construct a Recorder
directly (no dsn) are completely unaffected. Postgres rows carry an extra
`asset` column (RecorderConfig.asset) since a shared database can no longer
rely on "which file" to disambiguate BTC/ETH/SOL the way SQLite did.

connect()/close() are deliberately kept SYNCHRONOUS (many tests call them
without awaiting, outside an event loop) -- the actual asyncpg pool is
created and torn down inside _writer_loop(), which is the one place
guaranteed to run inside the event loop as a real asyncio task.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

from .config import RecorderConfig

log = logging.getLogger(__name__)

_BOOK_LEVELS = 10


def _level_columns(prefix: str, real_type: str = "REAL") -> str:
    cols = []
    for i in range(1, _BOOK_LEVELS + 1):
        cols.append(f"{prefix}_{i}_price {real_type}, {prefix}_{i}_size {real_type}")
    return ",\n    ".join(cols)


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS markets (
    market_id TEXT PRIMARY KEY,
    slug TEXT,
    condition_id TEXT,
    up_token_id TEXT,
    down_token_id TEXT,
    start_ts INTEGER,
    end_ts INTEGER,
    tick_size REAL,
    reference_price REAL,
    resolution_source TEXT,
    twap_window INTEGER,
    maker_base_fee REAL,
    taker_base_fee REAL,
    fee_rate REAL,
    fee_exponent REAL,
    created_at INTEGER
);

CREATE TABLE IF NOT EXISTS market_settlement (
    market_id TEXT PRIMARY KEY,
    official_outcome TEXT,
    resolution_ts INTEGER,
    derived_outcome TEXT,
    derived_twap_at_end REAL,
    reference_price REAL,
    checked_at INTEGER
);

-- Live deterministic signal monitor (optional; reference implementation
-- only, NOT the ground truth for research -- discovery.py replays
-- arbitrary signal configs over the raw tables below instead). The
-- collector entry point no longer populates this (see main.py's
-- 2026-08-25 collector-only simplification); kept in the schema for any
-- ad-hoc/manual use.
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    market_id TEXT,
    direction TEXT,
    signal_ts INTEGER,
    z_1s REAL, momentum_250ms REAL, flow_1s REAL, imbalance REAL,
    poly_ask_signal REAL, poly_bid_signal REAL, poly_spread_signal REAL,
    remaining_s REAL, ask_change_500ms REAL,
    is_debug INTEGER
);
CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market_id);

-- ---- Raw data: source of truth ----

CREATE TABLE IF NOT EXISTS binance_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT,
    agg_trade_id INTEGER,
    exchange_ts INTEGER,
    recv_ts INTEGER,
    recv_monotonic_ns INTEGER,
    price REAL,
    qty REAL,
    aggressor_side TEXT,
    is_debug INTEGER
);
CREATE INDEX IF NOT EXISTS idx_binance_trades_ts ON binance_trades(exchange_ts);
CREATE INDEX IF NOT EXISTS idx_binance_trades_market ON binance_trades(market_id, exchange_ts);

CREATE TABLE IF NOT EXISTS binance_book (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT,
    exchange_ts INTEGER,
    recv_ts INTEGER,
    recv_monotonic_ns INTEGER,
    best_bid_price REAL, best_bid_size REAL,
    best_ask_price REAL, best_ask_size REAL,
    {_level_columns("bid")},
    {_level_columns("ask")},
    is_debug INTEGER
);
CREATE INDEX IF NOT EXISTS idx_binance_book_ts ON binance_book(recv_ts);
CREATE INDEX IF NOT EXISTS idx_binance_book_market ON binance_book(market_id, recv_ts);

CREATE TABLE IF NOT EXISTS polymarket_book (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT,
    ts INTEGER,
    recv_monotonic_ns INTEGER,
    up_best_bid REAL, up_best_bid_size REAL, up_best_ask REAL, up_best_ask_size REAL,
    {_level_columns("up_bid")},
    {_level_columns("up_ask")},
    down_best_bid REAL, down_best_bid_size REAL, down_best_ask REAL, down_best_ask_size REAL,
    {_level_columns("down_bid")},
    {_level_columns("down_ask")},
    is_debug INTEGER
);
CREATE INDEX IF NOT EXISTS idx_polymarket_book_market_ts ON polymarket_book(market_id, ts);

CREATE TABLE IF NOT EXISTS chainlink_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT,
    observation_ts INTEGER,
    recv_ts INTEGER,
    recv_monotonic_ns INTEGER,
    twap_value REAL,
    twap_window INTEGER,
    is_debug INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chainlink_obs_ts ON chainlink_observations(observation_ts);
CREATE INDEX IF NOT EXISTS idx_chainlink_obs_market ON chainlink_observations(market_id, observation_ts);

-- ---- Convenience only: always re-derivable from the raw tables above ----

CREATE TABLE IF NOT EXISTS binance_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER,
    market_id TEXT,
    btc_mid REAL,
    return_100ms REAL, return_250ms REAL, return_500ms REAL,
    return_1s REAL, return_2s REAL, return_5s REAL,
    vol_10s REAL, vol_30s REAL, vol_60s REAL, vol_120s REAL,
    flow_250ms REAL, flow_500ms REAL, flow_1s REAL, flow_2s REAL, flow_5s REAL,
    imbalance_top1 REAL, imbalance_top5 REAL, imbalance_top10 REAL,
    is_debug INTEGER
);
CREATE INDEX IF NOT EXISTS idx_binance_features_market_ts ON binance_features(market_id, ts);
"""


# ---------------------------------------------------------------------------
# Postgres schema: same tables, Postgres-native types (BIGINT for every ms
# timestamp -- SQLite's dynamic typing silently allowed 64-bit values in an
# "INTEGER" column, Postgres would truncate/reject them under plain
# INTEGER's 32-bit range), BIGSERIAL for autoincrement, and an explicit
# `asset` column on every table since one Postgres database is shared by
# all three collectors (see module docstring).
# ---------------------------------------------------------------------------

def _pg_level_columns(prefix: str) -> str:
    return _level_columns(prefix, real_type="DOUBLE PRECISION")


PG_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS markets (
    market_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    slug TEXT,
    condition_id TEXT,
    up_token_id TEXT,
    down_token_id TEXT,
    start_ts BIGINT,
    end_ts BIGINT,
    tick_size DOUBLE PRECISION,
    reference_price DOUBLE PRECISION,
    resolution_source TEXT,
    twap_window BIGINT,
    maker_base_fee DOUBLE PRECISION,
    taker_base_fee DOUBLE PRECISION,
    fee_rate DOUBLE PRECISION,
    fee_exponent DOUBLE PRECISION,
    created_at BIGINT
);

CREATE TABLE IF NOT EXISTS market_settlement (
    market_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    official_outcome TEXT,
    resolution_ts BIGINT,
    derived_outcome TEXT,
    derived_twap_at_end DOUBLE PRECISION,
    reference_price DOUBLE PRECISION,
    checked_at BIGINT
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    market_id TEXT,
    direction TEXT,
    signal_ts BIGINT,
    z_1s DOUBLE PRECISION, momentum_250ms DOUBLE PRECISION, flow_1s DOUBLE PRECISION,
    imbalance DOUBLE PRECISION,
    poly_ask_signal DOUBLE PRECISION, poly_bid_signal DOUBLE PRECISION,
    poly_spread_signal DOUBLE PRECISION,
    remaining_s DOUBLE PRECISION, ask_change_500ms DOUBLE PRECISION,
    is_debug SMALLINT
);
CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market_id);

-- ---- Raw data: source of truth ----

CREATE TABLE IF NOT EXISTS binance_trades (
    id BIGSERIAL PRIMARY KEY,
    asset TEXT NOT NULL,
    market_id TEXT,
    agg_trade_id BIGINT,
    exchange_ts BIGINT,
    recv_ts BIGINT,
    recv_monotonic_ns BIGINT,
    price DOUBLE PRECISION,
    qty DOUBLE PRECISION,
    aggressor_side TEXT,
    is_debug SMALLINT
);
CREATE INDEX IF NOT EXISTS idx_binance_trades_ts ON binance_trades(exchange_ts);
CREATE INDEX IF NOT EXISTS idx_binance_trades_market ON binance_trades(market_id, exchange_ts);

CREATE TABLE IF NOT EXISTS binance_book (
    id BIGSERIAL PRIMARY KEY,
    asset TEXT NOT NULL,
    market_id TEXT,
    exchange_ts BIGINT,
    recv_ts BIGINT,
    recv_monotonic_ns BIGINT,
    best_bid_price DOUBLE PRECISION, best_bid_size DOUBLE PRECISION,
    best_ask_price DOUBLE PRECISION, best_ask_size DOUBLE PRECISION,
    {_pg_level_columns("bid")},
    {_pg_level_columns("ask")},
    is_debug SMALLINT
);
CREATE INDEX IF NOT EXISTS idx_binance_book_ts ON binance_book(recv_ts);
CREATE INDEX IF NOT EXISTS idx_binance_book_market ON binance_book(market_id, recv_ts);

CREATE TABLE IF NOT EXISTS polymarket_book (
    id BIGSERIAL PRIMARY KEY,
    asset TEXT NOT NULL,
    market_id TEXT,
    ts BIGINT,
    recv_monotonic_ns BIGINT,
    up_best_bid DOUBLE PRECISION, up_best_bid_size DOUBLE PRECISION,
    up_best_ask DOUBLE PRECISION, up_best_ask_size DOUBLE PRECISION,
    {_pg_level_columns("up_bid")},
    {_pg_level_columns("up_ask")},
    down_best_bid DOUBLE PRECISION, down_best_bid_size DOUBLE PRECISION,
    down_best_ask DOUBLE PRECISION, down_best_ask_size DOUBLE PRECISION,
    {_pg_level_columns("down_bid")},
    {_pg_level_columns("down_ask")},
    is_debug SMALLINT
);
CREATE INDEX IF NOT EXISTS idx_polymarket_book_market_ts ON polymarket_book(market_id, ts);

CREATE TABLE IF NOT EXISTS chainlink_observations (
    id BIGSERIAL PRIMARY KEY,
    asset TEXT NOT NULL,
    market_id TEXT,
    observation_ts BIGINT,
    recv_ts BIGINT,
    recv_monotonic_ns BIGINT,
    twap_value DOUBLE PRECISION,
    twap_window BIGINT,
    is_debug SMALLINT
);
CREATE INDEX IF NOT EXISTS idx_chainlink_obs_ts ON chainlink_observations(observation_ts);
CREATE INDEX IF NOT EXISTS idx_chainlink_obs_market ON chainlink_observations(market_id, observation_ts);

-- ---- Convenience only: always re-derivable from the raw tables above ----

CREATE TABLE IF NOT EXISTS binance_features (
    id BIGSERIAL PRIMARY KEY,
    asset TEXT NOT NULL,
    ts BIGINT,
    market_id TEXT,
    btc_mid DOUBLE PRECISION,
    return_100ms DOUBLE PRECISION, return_250ms DOUBLE PRECISION, return_500ms DOUBLE PRECISION,
    return_1s DOUBLE PRECISION, return_2s DOUBLE PRECISION, return_5s DOUBLE PRECISION,
    vol_10s DOUBLE PRECISION, vol_30s DOUBLE PRECISION, vol_60s DOUBLE PRECISION, vol_120s DOUBLE PRECISION,
    flow_250ms DOUBLE PRECISION, flow_500ms DOUBLE PRECISION, flow_1s DOUBLE PRECISION,
    flow_2s DOUBLE PRECISION, flow_5s DOUBLE PRECISION,
    imbalance_top1 DOUBLE PRECISION, imbalance_top5 DOUBLE PRECISION, imbalance_top10 DOUBLE PRECISION,
    is_debug SMALLINT
);
CREATE INDEX IF NOT EXISTS idx_binance_features_market_ts ON binance_features(market_id, ts);
"""

# Tables with a real business-key primary key get an upsert (ON CONFLICT DO
# UPDATE) -- markets/settlement rows are legitimately re-written as more
# info arrives, and signals could in principle be re-sent. The append-only
# event tables (binance_trades, binance_book, polymarket_book,
# chainlink_observations, binance_features) use a surrogate BIGSERIAL id
# that never collides, so a plain INSERT is correct and cheaper for them.
PG_UPSERT_KEYS: dict[str, str] = {
    "markets": "market_id",
    "market_settlement": "market_id",
    "signals": "signal_id",
}


def _pg_insert_sql(table: str, columns: list[str]) -> str:
    placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
    cols_sql = ", ".join(columns)
    pk = PG_UPSERT_KEYS.get(table)
    if pk is None:
        return f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders})"
    update_cols = [c for c in columns if c != pk]
    if not update_cols:
        return f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) ON CONFLICT ({pk}) DO NOTHING"
    set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
    return f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) ON CONFLICT ({pk}) DO UPDATE SET {set_clause}"


@dataclass
class WriteJob:
    table: str
    row: dict[str, Any]


# Caps how much unwritten data can pile up in memory if the writer stalls
# (network hiccup to Postgres, a hung connection, pool exhaustion...). A
# real production incident: an unbounded queue here let a stalled Postgres
# write silently balloon memory for ~40 minutes before anyone noticed --
# the WebSocket feeds kept receiving fine, only the write side was stuck,
# so nothing crashed or logged an error, it just grew. Past this many
# unwritten rows, enqueue() drops the new row (loud, rate-limited log)
# rather than let memory grow without bound -- losing a slice of raw data
# during a genuine outage is recoverable; an OOM kill losing the entire
# in-memory queue is not.
MAX_QUEUE_SIZE = 50_000
# A single batch write must complete within this long or it's abandoned --
# without this, one hung `await conn.execute(...)` (dead connection the
# pool didn't detect, a network black hole) blocks the writer loop
# forever, which is exactly what turns a transient hiccup into the
# unbounded-queue-growth scenario above.
PG_WRITE_TIMEOUT_S = 15.0


class Recorder:
    def __init__(self, cfg: RecorderConfig):
        self.cfg = cfg
        self._queue: asyncio.Queue[WriteJob] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._conn: sqlite3.Connection | None = None
        self._task: asyncio.Task | None = None
        self._pg_pool: Any = None  # asyncpg.Pool, created lazily inside _writer_loop()
        self._dropped_since_log = 0

    def connect(self) -> None:
        if self.cfg.dsn:
            # Postgres: pool creation needs `await` and must happen inside
            # the event loop, so it's deferred to _writer_loop()'s first
            # iteration (the one place guaranteed to run as a real asyncio
            # task) -- nothing to do synchronously here.
            return
        os.makedirs(os.path.dirname(self.cfg.db_path) or ".", exist_ok=True)
        # check_same_thread=False: writes happen via asyncio.to_thread, which
        # may run on a different worker thread each call. Safe here because
        # the writer loop awaits each batch fully before starting the next
        # (never truly concurrent access).
        self._conn = sqlite3.connect(self.cfg.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def start(self) -> None:
        self._task = asyncio.create_task(self._writer_loop())

    def enqueue(self, table: str, row: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(WriteJob(table, row))
        except asyncio.QueueFull:
            # Rate-limited: sustained backpressure calls this on every
            # single event, one log line per drop would itself add load.
            self._dropped_since_log += 1
            if self._dropped_since_log == 1 or self._dropped_since_log % 1_000 == 0:
                log.error("write queue full (%d), dropped %d row(s) for %s so far -- writer is stalled",
                          MAX_QUEUE_SIZE, self._dropped_since_log, table)

    async def _writer_loop(self) -> None:
        if self.cfg.dsn:
            import asyncpg  # local import: only required when Postgres is actually used

            self._pg_pool = await asyncpg.create_pool(self.cfg.dsn, min_size=1, max_size=4)
            async with self._pg_pool.acquire() as conn:
                # asyncpg's simple query protocol (no args) accepts multiple
                # ;-separated statements in one call, same shape as
                # sqlite3.executescript() above.
                await conn.execute(PG_SCHEMA)
            log.info("connected to Postgres, schema ensured (asset=%s)", self.cfg.asset)

        while True:
            job = await self._queue.get()
            batch = [job]
            try:
                while len(batch) < self.cfg.write_batch_size:
                    batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                pass
            if self.cfg.dsn:
                try:
                    await asyncio.wait_for(self._write_batch_pg(batch), timeout=PG_WRITE_TIMEOUT_S)
                except asyncio.TimeoutError:
                    log.error("Postgres batch write timed out after %.0fs (%d rows abandoned) -- "
                              "writer loop continuing, not blocking forever", PG_WRITE_TIMEOUT_S, len(batch))
            else:
                await asyncio.to_thread(self._write_batch, batch)
            for _ in batch:
                self._queue.task_done()
            if self._dropped_since_log and self._queue.qsize() < MAX_QUEUE_SIZE // 2:
                log.warning("write queue recovered after dropping %d row(s)", self._dropped_since_log)
                self._dropped_since_log = 0

    def _write_batch(self, batch: list[WriteJob]) -> None:
        assert self._conn is not None
        for job in batch:
            cols = ", ".join(job.row.keys())
            placeholders = ", ".join(f":{k}" for k in job.row.keys())
            sql = f"INSERT OR REPLACE INTO {job.table} ({cols}) VALUES ({placeholders})"
            try:
                self._conn.execute(sql, job.row)
            except sqlite3.Error:
                log.exception("failed to write row to %s", job.table)
        self._conn.commit()

    async def _write_batch_pg(self, batch: list[WriteJob]) -> None:
        assert self._pg_pool is not None
        async with self._pg_pool.acquire() as conn:
            async with conn.transaction():
                for job in batch:
                    row = dict(job.row)
                    row.setdefault("asset", self.cfg.asset)
                    columns = list(row.keys())
                    sql = _pg_insert_sql(job.table, columns)
                    try:
                        await conn.execute(sql, *row.values())
                    except Exception:
                        log.exception("failed to write row to %s", job.table)

    async def flush(self) -> None:
        await self._queue.join()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        # Postgres pool: NOT closed here -- close() is called synchronously
        # (including from non-async test code), and asyncpg has no sync
        # close path. Best-effort cleanup on process exit is acceptable for
        # a long-running collector that's stopped via SIGINT/SIGTERM; a
        # clean `await pool.close()` would need an async shutdown path
        # threaded through main.py's run() if ever needed.
