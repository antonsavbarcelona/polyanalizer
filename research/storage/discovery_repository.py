"""SQLite repository for the signal-response discovery pipeline. A
SEPARATE database/schema from research.storage.results_repository (the
TP/SL pipeline's) -- contract #1 forbids this pipeline from ever mixing
with TP/SL concepts, and a shared DB file would make that boundary easy to
blur by accident.

Single writer (contract #39): this class is only ever called from the
orchestrator's parent process. Worker processes compute and return plain
result bundles; they never touch this repository directly.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import asdict
from typing import Any

from research.discovery_types import (
    ALL_ASSET,
    LEVELS,
    Control,
    ControlEntryMark,
    ControlResponse,
    DiscoveryEntryMark,
    SignalConfigSummary,
    SignalDiscoveryConfig,
    SignalFirstPassageSummary,
    SignalHitSummary,
    SignalPathStats,
    SignalResponse,
    SignalResponseSummary,
    SignalSnapshot,
    canonical_json,
    deterministic_id,
    level_field_suffix,
)


def _level_columns(prefix: str) -> str:
    return ",\n    ".join(f"{prefix}_{level_field_suffix(lvl)}_ms INTEGER" for lvl in LEVELS)


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    error_json TEXT,
    config_json TEXT NOT NULL,
    data_sources TEXT NOT NULL,
    code_version TEXT,
    data_fingerprint TEXT,
    fee_model_version TEXT,
    feature_version TEXT,
    control_matching_version TEXT,
    response_engine_version TEXT,
    bootstrap_seed INTEGER,
    bootstrap_iterations INTEGER,
    volatility_boundaries_json TEXT
);

CREATE TABLE IF NOT EXISTS signal_configs (
    signal_config_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL
);

-- Grain is (experiment, signal_config, asset) -- task requirement: if BTC
-- and ETH complete but SOL fails, --resume must only recompute SOL, never
-- redo the two assets that already finished.
CREATE TABLE IF NOT EXISTS checkpoints (
    experiment_id TEXT NOT NULL,
    signal_config_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    status TEXT NOT NULL,  -- PENDING | RUNNING | COMPLETE | FAILED
    started_at INTEGER,
    completed_at INTEGER,
    error TEXT,
    PRIMARY KEY (experiment_id, signal_config_id, asset)
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    signal_config_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    market_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    signal_ts INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    UNIQUE (signal_config_id, asset, market_id, signal_ts, direction)
);
CREATE INDEX IF NOT EXISTS idx_signals_config ON signals(experiment_id, signal_config_id);
CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market_id);

CREATE TABLE IF NOT EXISTS entry_marks (
    entry_mark_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    size_shares REAL NOT NULL,
    entry_target_ts INTEGER NOT NULL,
    entry_actual_ts INTEGER,
    entry_delay_after_target_ms INTEGER,
    entry_best_bid REAL,
    entry_best_ask REAL,
    entry_vwap REAL,
    entry_slippage REAL,
    available_ask_liquidity REAL,
    entry_fee_total REAL,
    entry_fee_per_share REAL,
    status TEXT NOT NULL,
    UNIQUE (signal_id, latency_ms, size_shares)
);
CREATE INDEX IF NOT EXISTS idx_entry_marks_signal ON entry_marks(signal_id);

CREATE TABLE IF NOT EXISTS signal_response (
    response_id TEXT PRIMARY KEY,
    entry_mark_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    signal_config_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    market_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    size_shares REAL NOT NULL,
    horizon_ms INTEGER NOT NULL,
    response_target_ts INTEGER NOT NULL,
    response_actual_ts INTEGER,
    response_delay_ms INTEGER,
    future_best_bid REAL,
    future_best_ask REAL,
    future_sell_vwap REAL,
    available_bid_liquidity REAL,
    raw_response REAL,
    fee_adjusted_response REAL,
    response_positive INTEGER,
    fee_adjusted_positive INTEGER,
    status TEXT NOT NULL,
    UNIQUE (entry_mark_id, horizon_ms)
);
CREATE INDEX IF NOT EXISTS idx_signal_response_config ON signal_response(signal_config_id, asset, latency_ms, size_shares, horizon_ms);

CREATE TABLE IF NOT EXISTS signal_path_stats (
    entry_mark_id TEXT NOT NULL,
    stats_horizon_ms INTEGER NOT NULL,
    mfe REAL,
    mae REAL,
    time_to_mfe_ms INTEGER,
    time_to_mae_ms INTEGER,
    {_level_columns("time_to_plus")},
    {_level_columns("time_to_minus")},
    PRIMARY KEY (entry_mark_id, stats_horizon_ms)
);

CREATE TABLE IF NOT EXISTS controls (
    control_id TEXT PRIMARY KEY,
    source_signal_id TEXT NOT NULL,
    signal_config_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    market_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    control_ts INTEGER NOT NULL,
    match_rank INTEGER NOT NULL,
    match_distance REAL NOT NULL,
    tte_regime TEXT,
    price_regime TEXT,
    spread_regime TEXT,
    volatility_regime TEXT
);
CREATE INDEX IF NOT EXISTS idx_controls_source ON controls(source_signal_id);

CREATE TABLE IF NOT EXISTS control_entry_marks (
    control_entry_mark_id TEXT PRIMARY KEY,
    control_id TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    size_shares REAL NOT NULL,
    entry_target_ts INTEGER NOT NULL,
    entry_actual_ts INTEGER,
    entry_delay_after_target_ms INTEGER,
    entry_best_bid REAL,
    entry_best_ask REAL,
    entry_vwap REAL,
    entry_slippage REAL,
    available_ask_liquidity REAL,
    entry_fee_total REAL,
    entry_fee_per_share REAL,
    status TEXT NOT NULL,
    UNIQUE (control_id, latency_ms, size_shares)
);

CREATE TABLE IF NOT EXISTS control_response (
    control_response_id TEXT PRIMARY KEY,
    control_entry_mark_id TEXT NOT NULL,
    control_id TEXT NOT NULL,
    signal_config_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    market_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    size_shares REAL NOT NULL,
    horizon_ms INTEGER NOT NULL,
    response_target_ts INTEGER NOT NULL,
    response_actual_ts INTEGER,
    response_delay_ms INTEGER,
    future_best_bid REAL,
    future_best_ask REAL,
    future_sell_vwap REAL,
    available_bid_liquidity REAL,
    raw_response REAL,
    fee_adjusted_response REAL,
    response_positive INTEGER,
    fee_adjusted_positive INTEGER,
    status TEXT NOT NULL,
    UNIQUE (control_entry_mark_id, horizon_ms)
);
CREATE INDEX IF NOT EXISTS idx_control_response_config ON control_response(signal_config_id, asset, latency_ms, size_shares, horizon_ms);

CREATE TABLE IF NOT EXISTS signal_config_summary (
    experiment_id TEXT NOT NULL,
    signal_config_id TEXT NOT NULL,
    asset TEXT,
    latency_ms INTEGER NOT NULL,
    size_shares REAL NOT NULL,
    signal_count INTEGER NOT NULL,
    market_count INTEGER NOT NULL,
    executable_entry_count INTEGER NOT NULL,
    not_executable_entry_count INTEGER NOT NULL,
    entry_execution_rate REAL,
    mean_entry_delay_ms REAL,
    median_entry_delay_ms REAL,
    mean_entry_slippage REAL,
    median_entry_slippage REAL,
    control_count INTEGER NOT NULL,
    plateau_score REAL,
    neighbor_count INTEGER,
    neighbor_positive_ratio REAL,
    neighbor_mean_uplift REAL,
    neighbor_std_uplift REAL,
    PRIMARY KEY (experiment_id, signal_config_id, asset, latency_ms, size_shares)
);

CREATE TABLE IF NOT EXISTS signal_response_summary (
    experiment_id TEXT NOT NULL,
    signal_config_id TEXT NOT NULL,
    asset TEXT,
    latency_ms INTEGER NOT NULL,
    size_shares REAL NOT NULL,
    horizon_ms INTEGER NOT NULL,
    available_count INTEGER NOT NULL,
    unavailable_count INTEGER NOT NULL,
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    zero_count INTEGER NOT NULL,
    p_positive REAL,
    mean_response REAL,
    median_response REAL,
    std_response REAL,
    p05 REAL, p10 REAL, p25 REAL, p50 REAL, p75 REAL, p90 REAL, p95 REAL,
    mean_fee_adjusted_response REAL,
    median_fee_adjusted_response REAL,
    control_available_count INTEGER NOT NULL,
    control_p_positive REAL,
    control_mean_response REAL,
    control_median_response REAL,
    uplift_p_positive REAL,
    uplift_mean_response REAL,
    uplift_median_response REAL,
    bootstrap_ci95_mean_low REAL,
    bootstrap_ci95_mean_high REAL,
    bootstrap_ci95_p_positive_low REAL,
    bootstrap_ci95_p_positive_high REAL,
    bootstrap_ci95_uplift_low REAL,
    bootstrap_ci95_uplift_high REAL,
    PRIMARY KEY (experiment_id, signal_config_id, asset, latency_ms, size_shares, horizon_ms)
);

CREATE TABLE IF NOT EXISTS signal_hit_summary (
    experiment_id TEXT NOT NULL,
    signal_config_id TEXT NOT NULL,
    asset TEXT,
    latency_ms INTEGER NOT NULL,
    size_shares REAL NOT NULL,
    horizon_ms INTEGER NOT NULL,
    level REAL NOT NULL,
    direction TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    hit_count INTEGER NOT NULL,
    hit_probability REAL,
    PRIMARY KEY (experiment_id, signal_config_id, asset, latency_ms, size_shares, horizon_ms, level, direction)
);

CREATE TABLE IF NOT EXISTS signal_first_passage_summary (
    experiment_id TEXT NOT NULL,
    signal_config_id TEXT NOT NULL,
    asset TEXT,
    latency_ms INTEGER NOT NULL,
    size_shares REAL NOT NULL,
    horizon_ms INTEGER NOT NULL,
    plus_level REAL NOT NULL,
    minus_level REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    plus_first_count INTEGER NOT NULL,
    minus_first_count INTEGER NOT NULL,
    plus_only_count INTEGER NOT NULL,
    minus_only_count INTEGER NOT NULL,
    ambiguous_count INTEGER NOT NULL,
    neither_count INTEGER NOT NULL,
    p_plus_first_given_hit REAL,
    p_minus_first_given_hit REAL,
    PRIMARY KEY (experiment_id, signal_config_id, asset, latency_ms, size_shares, horizon_ms, plus_level, minus_level)
);
"""


class DiscoveryRepository:
    def __init__(self, db_path: str = "data/signal_discovery_results.db"):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "DiscoveryRepository":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_conn(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("DiscoveryRepository.connect() must be called first")
        return self.conn

    # ---- experiment lifecycle ----

    def create_experiment(self, name: str, config: Any, *, data_sources: Any, data_fingerprint: str | None,
                           code_version: str | None, fee_model_version: str, bootstrap_seed: int,
                           bootstrap_iterations: int, feature_version: str = "v1",
                           control_matching_version: str = "v1", response_engine_version: str = "v1",
                           experiment_id: str | None = None) -> str:
        """feature_version/control_matching_version/response_engine_version
        are frozen into every experiment row so a future change to any of
        these algorithms can never get silently mixed with results computed
        under the old one -- compare these before comparing numbers across
        experiment_ids."""
        conn = self._require_conn()
        started_at = int(time.time() * 1000)
        payload = {"name": name, "config": config, "data_sources": data_sources}
        experiment_id = experiment_id or deterministic_id("sr_experiment", payload)
        conn.execute(
            """INSERT OR REPLACE INTO experiments
               (experiment_id, name, started_at, finished_at, status, error_json, config_json, data_sources,
                code_version, data_fingerprint, fee_model_version, feature_version, control_matching_version,
                response_engine_version, bootstrap_seed, bootstrap_iterations, volatility_boundaries_json)
               VALUES (?, ?, ?, NULL, 'RUNNING', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (experiment_id, name, started_at, canonical_json(config), canonical_json(data_sources),
             code_version, data_fingerprint, fee_model_version, feature_version, control_matching_version,
             response_engine_version, bootstrap_seed, bootstrap_iterations),
        )
        conn.commit()
        return experiment_id

    def set_volatility_boundaries(self, experiment_id: str, boundaries_json: str) -> None:
        conn = self._require_conn()
        conn.execute("UPDATE experiments SET volatility_boundaries_json=? WHERE experiment_id=?",
                      (boundaries_json, experiment_id))
        conn.commit()

    def finish_experiment(self, experiment_id: str, status: str, error: Any = None) -> None:
        conn = self._require_conn()
        conn.execute(
            "UPDATE experiments SET status=?, finished_at=?, error_json=? WHERE experiment_id=?",
            (status, int(time.time() * 1000), canonical_json(error) if error is not None else None, experiment_id),
        )
        conn.commit()

    # ---- checkpoints (task: grain is experiment x signal_config x asset,
    # so BTC/ETH COMPLETE + SOL FAILED only ever recomputes SOL) ----

    def checkpoint_status(self, experiment_id: str, signal_config_id: str, asset: str) -> str | None:
        conn = self._require_conn()
        row = conn.execute(
            "SELECT status FROM checkpoints WHERE experiment_id=? AND signal_config_id=? AND asset=?",
            (experiment_id, signal_config_id, asset),
        ).fetchone()
        return row["status"] if row else None

    def asset_checkpoint_statuses(self, experiment_id: str, signal_config_id: str) -> dict[str, str]:
        conn = self._require_conn()
        return {
            r["asset"]: r["status"]
            for r in conn.execute(
                "SELECT asset, status FROM checkpoints WHERE experiment_id=? AND signal_config_id=?",
                (experiment_id, signal_config_id),
            )
        }

    def mark_checkpoint_running(self, experiment_id: str, signal_config_id: str, asset: str) -> None:
        conn = self._require_conn()
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (experiment_id, signal_config_id, asset, status, started_at) "
            "VALUES (?, ?, ?, 'RUNNING', ?)",
            (experiment_id, signal_config_id, asset, int(time.time() * 1000)),
        )
        conn.commit()

    def mark_checkpoint_complete(self, experiment_id: str, signal_config_id: str, asset: str) -> None:
        # INSERT OR REPLACE, not UPDATE: discard_partial_asset (or a fresh
        # dispatch that never had a RUNNING row written) may leave no
        # existing checkpoint row to UPDATE, which would silently no-op.
        conn = self._require_conn()
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (experiment_id, signal_config_id, asset, status, completed_at, error) "
            "VALUES (?, ?, ?, 'COMPLETE', ?, NULL)",
            (experiment_id, signal_config_id, asset, int(time.time() * 1000)),
        )
        conn.commit()

    def mark_checkpoint_failed(self, experiment_id: str, signal_config_id: str, asset: str, error: str) -> None:
        conn = self._require_conn()
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (experiment_id, signal_config_id, asset, status, completed_at, error) "
            "VALUES (?, ?, ?, 'FAILED', ?, ?)",
            (experiment_id, signal_config_id, asset, int(time.time() * 1000), error),
        )
        conn.commit()

    def discard_partial_asset(self, experiment_id: str, signal_config_id: str, asset: str) -> None:
        """--resume must never mix a half-written (config, asset) with
        fresh data: wipe every row it might have produced before
        recomputing it from scratch. Only this asset's rows are touched --
        other, already-COMPLETE assets of the same config are untouched.
        The pooled (asset=ALL_ASSET) summary rows for this config are also
        cleared since they'd be stale once this asset's data changes; the
        pooling pass recomputes them once every asset is COMPLETE again."""
        conn = self._require_conn()
        signal_ids = [r["signal_id"] for r in conn.execute(
            "SELECT signal_id FROM signals WHERE experiment_id=? AND signal_config_id=? AND asset=?",
            (experiment_id, signal_config_id, asset),
        )]
        for signal_id in signal_ids:
            entry_mark_ids = [r["entry_mark_id"] for r in conn.execute(
                "SELECT entry_mark_id FROM entry_marks WHERE signal_id=?", (signal_id,))]
            for em_id in entry_mark_ids:
                conn.execute("DELETE FROM signal_response WHERE entry_mark_id=?", (em_id,))
                conn.execute("DELETE FROM signal_path_stats WHERE entry_mark_id=?", (em_id,))
            conn.execute("DELETE FROM entry_marks WHERE signal_id=?", (signal_id,))

            control_ids = [r["control_id"] for r in conn.execute(
                "SELECT control_id FROM controls WHERE source_signal_id=?", (signal_id,))]
            for control_id in control_ids:
                cem_ids = [r["control_entry_mark_id"] for r in conn.execute(
                    "SELECT control_entry_mark_id FROM control_entry_marks WHERE control_id=?", (control_id,))]
                for cem_id in cem_ids:
                    conn.execute("DELETE FROM control_response WHERE control_entry_mark_id=?", (cem_id,))
                conn.execute("DELETE FROM control_entry_marks WHERE control_id=?", (control_id,))
            conn.execute("DELETE FROM controls WHERE source_signal_id=?", (signal_id,))

        conn.execute("DELETE FROM signals WHERE experiment_id=? AND signal_config_id=? AND asset=?",
                     (experiment_id, signal_config_id, asset))
        conn.execute(
            "DELETE FROM signal_config_summary WHERE experiment_id=? AND signal_config_id=? AND asset=?",
            (experiment_id, signal_config_id, asset),
        )
        conn.execute(
            "DELETE FROM signal_response_summary WHERE experiment_id=? AND signal_config_id=? AND asset=?",
            (experiment_id, signal_config_id, asset),
        )
        conn.execute(
            "DELETE FROM signal_hit_summary WHERE experiment_id=? AND signal_config_id=? AND asset=?",
            (experiment_id, signal_config_id, asset),
        )
        conn.execute(
            "DELETE FROM signal_first_passage_summary WHERE experiment_id=? AND signal_config_id=? AND asset=?",
            (experiment_id, signal_config_id, asset),
        )
        # Pooled ALL rows (asset=ALL_ASSET) are now stale -- clear so a crash
        # right after this can't leave a stale ALL summary sitting around;
        # the pooling pass rebuilds them once every asset is COMPLETE.
        conn.execute(
            "DELETE FROM signal_config_summary WHERE experiment_id=? AND signal_config_id=? AND asset=?",
            (experiment_id, signal_config_id, ALL_ASSET),
        )
        conn.execute(
            "DELETE FROM signal_response_summary WHERE experiment_id=? AND signal_config_id=? AND asset=?",
            (experiment_id, signal_config_id, ALL_ASSET),
        )
        conn.execute(
            "DELETE FROM signal_hit_summary WHERE experiment_id=? AND signal_config_id=? AND asset=?",
            (experiment_id, signal_config_id, ALL_ASSET),
        )
        conn.execute(
            "DELETE FROM signal_first_passage_summary WHERE experiment_id=? AND signal_config_id=? AND asset=?",
            (experiment_id, signal_config_id, ALL_ASSET),
        )
        conn.execute("DELETE FROM checkpoints WHERE experiment_id=? AND signal_config_id=? AND asset=?",
                     (experiment_id, signal_config_id, asset))
        # No commit here -- callers batch this across many (config, asset)
        # pairs (see run_signal_discovery_experiment's dispatch-queue build)
        # and commit() once at the end; committing per-call was measured to
        # be a real bottleneck at hundreds/thousands of units.

    # ---- config registration ----
    #
    # NOTE on commits: every save_* method below deliberately does NOT call
    # conn.commit() -- a single (signal_config, asset) unit can produce tens
    # of thousands of rows (e.g. control_response: signals x controls x
    # latencies x horizons), and committing after every individual INSERT
    # was measured to dominate wall time (thousands of fsyncs where one
    # would do). Callers batch a whole unit's writes and call commit() ONCE
    # at the end (see research/discovery_experiment.py) -- checkpoint-level
    # granularity is the real unit of atomicity here, not per-row.

    def commit(self) -> None:
        self._require_conn().commit()

    def save_signal_config(self, cfg: SignalDiscoveryConfig) -> None:
        conn = self._require_conn()
        conn.execute(
            "INSERT OR REPLACE INTO signal_configs (signal_config_id, config_json, config_hash) VALUES (?, ?, ?)",
            (cfg.id, canonical_json(cfg), cfg.id),
        )

    # ---- rows ----

    def save_signal(self, experiment_id: str, snapshot: SignalSnapshot) -> None:
        conn = self._require_conn()
        conn.execute(
            """INSERT OR REPLACE INTO signals
               (signal_id, experiment_id, signal_config_id, asset, market_id, direction, signal_ts, snapshot_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (snapshot.signal_id, experiment_id, snapshot.signal_config_id, snapshot.asset, snapshot.market_id,
             snapshot.direction, snapshot.signal_ts, canonical_json(snapshot)),
        )

    def save_entry_mark(self, mark: DiscoveryEntryMark) -> None:
        conn = self._require_conn()
        data = asdict(mark)
        cols = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        conn.execute(f"INSERT OR REPLACE INTO entry_marks ({cols}) VALUES ({placeholders})", data)

    def save_signal_response(self, response: SignalResponse) -> None:
        conn = self._require_conn()
        data = asdict(response)
        data["response_positive"] = _bool_to_int(data["response_positive"])
        data["fee_adjusted_positive"] = _bool_to_int(data["fee_adjusted_positive"])
        cols = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        conn.execute(f"INSERT OR REPLACE INTO signal_response ({cols}) VALUES ({placeholders})", data)

    def save_path_stats(self, stats: SignalPathStats) -> None:
        conn = self._require_conn()
        data = {
            "entry_mark_id": stats.entry_mark_id, "stats_horizon_ms": stats.stats_horizon_ms,
            "mfe": stats.mfe, "mae": stats.mae,
            "time_to_mfe_ms": stats.time_to_mfe_ms, "time_to_mae_ms": stats.time_to_mae_ms,
        }
        for lvl in LEVELS:
            data[f"time_to_plus_{level_field_suffix(lvl)}_ms"] = stats.time_to_plus_ms.get(lvl)
            data[f"time_to_minus_{level_field_suffix(lvl)}_ms"] = stats.time_to_minus_ms.get(lvl)
        cols = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        conn.execute(f"INSERT OR REPLACE INTO signal_path_stats ({cols}) VALUES ({placeholders})", data)

    def save_control(self, control: Control) -> None:
        conn = self._require_conn()
        data = asdict(control)
        cols = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        conn.execute(f"INSERT OR REPLACE INTO controls ({cols}) VALUES ({placeholders})", data)

    def save_control_entry_mark(self, mark: ControlEntryMark) -> None:
        conn = self._require_conn()
        data = asdict(mark)
        cols = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        conn.execute(f"INSERT OR REPLACE INTO control_entry_marks ({cols}) VALUES ({placeholders})", data)

    def save_control_response(self, response: ControlResponse) -> None:
        conn = self._require_conn()
        data = asdict(response)
        data["response_positive"] = _bool_to_int(data["response_positive"])
        data["fee_adjusted_positive"] = _bool_to_int(data["fee_adjusted_positive"])
        cols = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        conn.execute(f"INSERT OR REPLACE INTO control_response ({cols}) VALUES ({placeholders})", data)

    def save_config_summary(self, experiment_id: str, summary: SignalConfigSummary, *,
                             neighbor_count: int | None = None, neighbor_positive_ratio: float | None = None,
                             neighbor_mean_uplift: float | None = None, neighbor_std_uplift: float | None = None
                             ) -> None:
        conn = self._require_conn()
        data = asdict(summary)
        data["experiment_id"] = experiment_id
        data["neighbor_count"] = neighbor_count
        data["neighbor_positive_ratio"] = neighbor_positive_ratio
        data["neighbor_mean_uplift"] = neighbor_mean_uplift
        data["neighbor_std_uplift"] = neighbor_std_uplift
        cols = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        conn.execute(f"INSERT OR REPLACE INTO signal_config_summary ({cols}) VALUES ({placeholders})", data)

    def save_response_summary(self, experiment_id: str, summary: SignalResponseSummary) -> None:
        conn = self._require_conn()
        data = asdict(summary)
        data["experiment_id"] = experiment_id
        cols = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        conn.execute(f"INSERT OR REPLACE INTO signal_response_summary ({cols}) VALUES ({placeholders})", data)

    def save_hit_summary(self, experiment_id: str, summary: SignalHitSummary) -> None:
        conn = self._require_conn()
        data = asdict(summary)
        data["experiment_id"] = experiment_id
        cols = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        conn.execute(f"INSERT OR REPLACE INTO signal_hit_summary ({cols}) VALUES ({placeholders})", data)

    def save_first_passage_summary(self, experiment_id: str, summary: SignalFirstPassageSummary) -> None:
        conn = self._require_conn()
        data = asdict(summary)
        data["experiment_id"] = experiment_id
        cols = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        conn.execute(f"INSERT OR REPLACE INTO signal_first_passage_summary ({cols}) VALUES ({placeholders})", data)

    # ---- reads (for CLI) ----

    def top_signal_configs(self, experiment_id: str, horizon_ms: int, latency_ms: int, size_shares: float,
                            limit: int = 20) -> list[sqlite3.Row]:
        """Canonical slice (asset=ALL, one latency, one size) -- without this
        filter a config appears once per latency_grid_ms value, which reads
        as duplicate leaderboard rows for "the same" config."""
        conn = self._require_conn()
        return conn.execute(
            """
            SELECT s.*, c.plateau_score, c.signal_count, c.market_count, c.entry_execution_rate
            FROM signal_response_summary s
            JOIN signal_config_summary c
              ON c.experiment_id=s.experiment_id AND c.signal_config_id=s.signal_config_id
             AND c.asset = s.asset AND c.latency_ms=s.latency_ms AND c.size_shares=s.size_shares
            WHERE s.experiment_id=? AND s.horizon_ms=? AND s.asset=?
              AND s.latency_ms=? AND s.size_shares=?
            ORDER BY COALESCE(c.plateau_score, -1e18) DESC
            LIMIT ?
            """,
            (experiment_id, horizon_ms, ALL_ASSET, latency_ms, size_shares, limit),
        ).fetchall()


def _bool_to_int(value: bool | None) -> int | None:
    return None if value is None else int(value)
