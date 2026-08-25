"""SQLite repository for reproducible research results."""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import asdict
from typing import Any

from poly_analyzer.discovery import DiscoveredSignal, ExecutionMark

from research.types import (
    BootstrapResult,
    ExecutionConfig,
    ExitConfig,
    SignalConfig,
    StrategyConfig,
    StrategyMetrics,
    TradeResult,
    canonical_json,
    deterministic_id,
)


SCHEMA = """
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
    fee_model TEXT,
    research_clock_ms INTEGER
);

CREATE TABLE IF NOT EXISTS signal_configs (
    signal_config_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_configs (
    execution_config_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exit_configs (
    exit_config_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    signal_config_id TEXT NOT NULL,
    execution_config_id TEXT NOT NULL,
    exit_config_id TEXT NOT NULL,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    experiment_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    signal_config_id TEXT NOT NULL,
    asset TEXT,
    market_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    signal_ts INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (experiment_id, signal_id)
);

CREATE TABLE IF NOT EXISTS entry_marks (
    experiment_id TEXT NOT NULL,
    entry_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    execution_config_id TEXT NOT NULL,
    requested_ts INTEGER NOT NULL,
    actual_ts INTEGER,
    best_ask REAL,
    entry_vwap REAL,
    filled_size REAL,
    entry_slippage REAL,
    status TEXT NOT NULL,
    PRIMARY KEY (experiment_id, entry_id)
);

CREATE TABLE IF NOT EXISTS trade_results (
    experiment_id TEXT NOT NULL,
    trade_result_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    asset TEXT,
    market_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    signal_ts INTEGER NOT NULL,
    entry_requested_ts INTEGER NOT NULL,
    entry_actual_ts INTEGER NOT NULL,
    entry_vwap REAL NOT NULL,
    exit_ts INTEGER,
    exit_price REAL,
    exit_reason TEXT NOT NULL,
    holding_ms INTEGER NOT NULL,
    gross_pnl_per_share REAL NOT NULL,
    entry_fee_per_share REAL,
    exit_fee_per_share REAL,
    fees_per_share REAL NOT NULL,
    net_pnl_per_share REAL NOT NULL,
    pnl_total REAL NOT NULL,
    mfe REAL,
    mae REAL,
    entry_slippage REAL,
    exit_slippage REAL,
    tp_hit INTEGER NOT NULL,
    sl_hit INTEGER NOT NULL,
    ambiguous_exit INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, trade_result_id)
);

CREATE TABLE IF NOT EXISTS strategy_metrics (
    experiment_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (experiment_id, strategy_id)
);

CREATE TABLE IF NOT EXISTS strategy_metrics_by_asset (
    experiment_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (experiment_id, strategy_id, asset)
);

CREATE TABLE IF NOT EXISTS strategy_metrics_by_market (
    experiment_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (experiment_id, strategy_id, market_id)
);

CREATE TABLE IF NOT EXISTS bootstrap_results (
    experiment_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    confidence REAL NOT NULL,
    low REAL,
    high REAL,
    iterations INTEGER NOT NULL,
    sampling_unit TEXT NOT NULL,
    PRIMARY KEY (experiment_id, strategy_id, metric, confidence, iterations, sampling_unit)
);
"""


class ResultsRepository:
    def __init__(self, db_path: str = "data/research_results.db"):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate_existing_tables()
        self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "ResultsRepository":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def create_experiment(
        self,
        name: str,
        config: Any,
        *,
        data_sources: Any = None,
        data_fingerprint: str | None = None,
        code_version: str | None = None,
        fee_model: str | None = None,
        research_clock_ms: int | None = None,
        started_at: int | None = None,
        experiment_id: str | None = None,
    ) -> str:
        conn = self._require_conn()
        started_at = started_at if started_at is not None else int(time.time() * 1000)
        payload = {
            "name": name,
            "config": config,
            "data_sources": data_sources,
            "data_fingerprint": data_fingerprint,
            "code_version": code_version,
            "fee_model": fee_model,
            "research_clock_ms": research_clock_ms,
        }
        experiment_id = experiment_id or deterministic_id("experiment", payload)
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments
            (experiment_id, name, started_at, finished_at, status, error_json,
             config_json, data_sources, code_version, data_fingerprint, fee_model, research_clock_ms)
            VALUES (?, ?, ?, NULL, 'RUNNING', NULL, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                name,
                started_at,
                canonical_json(config),
                canonical_json(data_sources or []),
                code_version,
                data_fingerprint,
                fee_model,
                research_clock_ms,
            ),
        )
        conn.commit()
        return experiment_id

    def finish_experiment(self, experiment_id: str, status: str, error: Any = None) -> None:
        conn = self._require_conn()
        conn.execute(
            """
            UPDATE experiments
            SET status=?, finished_at=?, error_json=?
            WHERE experiment_id=?
            """,
            (status, int(time.time() * 1000), canonical_json(error) if error is not None else None, experiment_id),
        )
        conn.commit()

    def save_signal_config(self, config: SignalConfig) -> str:
        return self._save_config("signal_configs", "signal_config_id", config.id, config)

    def save_execution_config(self, config: ExecutionConfig) -> str:
        return self._save_config("execution_configs", "execution_config_id", config.id, config)

    def save_exit_config(self, config: ExitConfig) -> str:
        return self._save_config("exit_configs", "exit_config_id", config.id, config)

    def save_strategy_config(self, config: StrategyConfig) -> str:
        conn = self._require_conn()
        signal_id = self.save_signal_config(config.signal)
        execution_id = self.save_execution_config(config.execution)
        exit_id = self.save_exit_config(config.exit)
        strategy_id = config.strategy_id()
        conn.execute(
            """
            INSERT OR REPLACE INTO strategies
            (strategy_id, signal_config_id, execution_config_id, exit_config_id, config_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (strategy_id, signal_id, execution_id, exit_id, canonical_json(config)),
        )
        conn.commit()
        return strategy_id

    def save_signal(self, experiment_id: str, signal: DiscoveredSignal, asset: str | None = None) -> str:
        conn = self._require_conn()
        signal_id = signal_identifier(signal)
        conn.execute(
            """
            INSERT OR REPLACE INTO signals
            (experiment_id, signal_id, signal_config_id, asset, market_id, direction, signal_ts, snapshot_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                signal_id,
                signal.config_id,
                asset,
                signal.market_id,
                signal.direction,
                signal.signal_ts,
                canonical_json(signal),
            ),
        )
        conn.commit()
        return signal_id

    def save_entry_mark(
        self,
        experiment_id: str,
        signal_id: str,
        execution_config_id: str,
        mark: ExecutionMark,
        status: str,
    ) -> str:
        conn = self._require_conn()
        entry_id = deterministic_id(
            "entry",
            {
                "signal_id": signal_id,
                "execution_config_id": execution_config_id,
                "requested_ts": mark.target_ts,
                "actual_ts": mark.actual_ts,
            },
        )
        entry_slippage = None
        if mark.entry_vwap is not None and mark.best_ask is not None:
            entry_slippage = mark.entry_vwap - mark.best_ask
        conn.execute(
            """
            INSERT OR REPLACE INTO entry_marks
            (experiment_id, entry_id, signal_id, execution_config_id, requested_ts, actual_ts,
             best_ask, entry_vwap, filled_size, entry_slippage, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                entry_id,
                signal_id,
                execution_config_id,
                mark.target_ts,
                mark.actual_ts,
                mark.best_ask,
                mark.entry_vwap,
                mark.filled_size,
                entry_slippage,
                status,
            ),
        )
        conn.commit()
        return entry_id

    def save_trade_result(self, result: TradeResult, experiment_id: str = "") -> str:
        conn = self._require_conn()
        result.trade_result_id = trade_identifier(result)
        data = asdict(result)
        data["experiment_id"] = experiment_id
        data["tp_hit"] = int(result.tp_hit)
        data["sl_hit"] = int(result.sl_hit)
        data["ambiguous_exit"] = int(result.ambiguous_exit)
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{key}" for key in data)
        conn.execute(f"INSERT OR REPLACE INTO trade_results ({columns}) VALUES ({placeholders})", data)
        conn.commit()
        return result.trade_result_id

    def save_strategy_metrics(self, metrics: StrategyMetrics, experiment_id: str = "") -> None:
        conn = self._require_conn()
        payload = canonical_json(metrics)
        if metrics.asset is not None and metrics.market_id is None:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_metrics_by_asset
                (experiment_id, strategy_id, asset, payload_json) VALUES (?, ?, ?, ?)
                """,
                (experiment_id, metrics.strategy_id, metrics.asset, payload),
            )
        elif metrics.market_id is not None and metrics.asset is None:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_metrics_by_market
                (experiment_id, strategy_id, market_id, payload_json) VALUES (?, ?, ?, ?)
                """,
                (experiment_id, metrics.strategy_id, metrics.market_id, payload),
            )
        else:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_metrics
                (experiment_id, strategy_id, payload_json) VALUES (?, ?, ?)
                """,
                (experiment_id, metrics.strategy_id, payload),
            )
        conn.commit()

    def save_bootstrap_result(self, result: BootstrapResult, experiment_id: str = "") -> None:
        conn = self._require_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO bootstrap_results
            (experiment_id, strategy_id, metric, confidence, low, high, iterations, sampling_unit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                result.strategy_id,
                result.metric,
                result.confidence,
                result.low,
                result.high,
                result.iterations,
                result.sampling_unit,
            ),
        )
        conn.commit()

    def leaderboard(self, experiment_id: str, limit: int = 20) -> list[sqlite3.Row]:
        conn = self._require_conn()
        return conn.execute(
            """
            SELECT m.strategy_id, m.payload_json,
                   COALESCE(bn.low, NULL) AS net_ci_low,
                   COALESCE(bn.high, NULL) AS net_ci_high,
                   COALESCE(bw.low, NULL) AS winrate_ci_low,
                   COALESCE(bw.high, NULL) AS winrate_ci_high,
                   s.config_json AS strategy_config_json
            FROM strategy_metrics m
            LEFT JOIN bootstrap_results bn
              ON bn.experiment_id=m.experiment_id AND bn.strategy_id=m.strategy_id AND bn.metric='mean_net_pnl'
            LEFT JOIN bootstrap_results bw
              ON bw.experiment_id=m.experiment_id AND bw.strategy_id=m.strategy_id AND bw.metric='win_rate'
            LEFT JOIN strategies s ON s.strategy_id=m.strategy_id
            WHERE m.experiment_id=?
            ORDER BY
              json_extract(m.payload_json, '$.mean_net_pnl') DESC,
              json_extract(m.payload_json, '$.trade_count') DESC,
              json_extract(m.payload_json, '$.profit_factor') DESC,
              m.strategy_id ASC
            LIMIT ?
            """,
            (experiment_id, limit),
        ).fetchall()

    def get_strategy_bundle(self, experiment_id: str, strategy_id: str) -> dict[str, Any]:
        conn = self._require_conn()
        return {
            "strategy": conn.execute("SELECT * FROM strategies WHERE strategy_id=?", (strategy_id,)).fetchone(),
            "metrics": conn.execute(
                "SELECT * FROM strategy_metrics WHERE experiment_id=? AND strategy_id=?",
                (experiment_id, strategy_id),
            ).fetchone(),
            "metrics_by_asset": conn.execute(
                "SELECT * FROM strategy_metrics_by_asset WHERE experiment_id=? AND strategy_id=? ORDER BY asset",
                (experiment_id, strategy_id),
            ).fetchall(),
            "metrics_by_market": conn.execute(
                "SELECT * FROM strategy_metrics_by_market WHERE experiment_id=? AND strategy_id=? ORDER BY market_id",
                (experiment_id, strategy_id),
            ).fetchall(),
            "bootstrap": conn.execute(
                "SELECT * FROM bootstrap_results WHERE experiment_id=? AND strategy_id=? ORDER BY metric",
                (experiment_id, strategy_id),
            ).fetchall(),
            "best_trades": conn.execute(
                """
                SELECT * FROM trade_results
                WHERE experiment_id=? AND strategy_id=?
                ORDER BY net_pnl_per_share DESC LIMIT 5
                """,
                (experiment_id, strategy_id),
            ).fetchall(),
            "worst_trades": conn.execute(
                """
                SELECT * FROM trade_results
                WHERE experiment_id=? AND strategy_id=?
                ORDER BY net_pnl_per_share ASC LIMIT 5
                """,
                (experiment_id, strategy_id),
            ).fetchall(),
        }

    def get_trade(self, trade_result_id: str, experiment_id: str | None = None) -> sqlite3.Row | None:
        conn = self._require_conn()
        if experiment_id is None:
            return conn.execute(
                "SELECT * FROM trade_results WHERE trade_result_id=?",
                (trade_result_id,),
            ).fetchone()
        return conn.execute(
            "SELECT * FROM trade_results WHERE experiment_id=? AND trade_result_id=?",
            (experiment_id, trade_result_id),
        ).fetchone()

    def _save_config(self, table: str, id_column: str, config_id: str, config: Any) -> str:
        conn = self._require_conn()
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({id_column}, config_json) VALUES (?, ?)",
            (config_id, canonical_json(config)),
        )
        conn.commit()
        return config_id

    def _migrate_existing_tables(self) -> None:
        assert self.conn is not None
        migrations = {
            "experiments": [
                ("finished_at", "INTEGER"),
                ("status", "TEXT NOT NULL DEFAULT 'RUNNING'"),
                ("error_json", "TEXT"),
                ("fee_model", "TEXT"),
                ("research_clock_ms", "INTEGER"),
            ],
            "trade_results": [
                ("experiment_id", "TEXT NOT NULL DEFAULT ''"),
                ("trade_result_id", "TEXT NOT NULL DEFAULT ''"),
                ("entry_fee_per_share", "REAL"),
                ("exit_fee_per_share", "REAL"),
                ("entry_slippage", "REAL"),
                ("exit_slippage", "REAL"),
            ],
            "strategy_metrics": [("experiment_id", "TEXT NOT NULL DEFAULT ''")],
            "strategy_metrics_by_asset": [("experiment_id", "TEXT NOT NULL DEFAULT ''")],
            "strategy_metrics_by_market": [("experiment_id", "TEXT NOT NULL DEFAULT ''")],
            "bootstrap_results": [("experiment_id", "TEXT NOT NULL DEFAULT ''")],
            "entry_marks": [
                ("experiment_id", "TEXT NOT NULL DEFAULT ''"),
                ("best_ask", "REAL"),
                ("filled_size", "REAL"),
                ("entry_slippage", "REAL"),
            ],
            "signals": [
                ("experiment_id", "TEXT NOT NULL DEFAULT ''"),
                ("signal_config_id", "TEXT NOT NULL DEFAULT ''"),
            ],
        }
        for table, columns in migrations.items():
            existing = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            for column, ddl in columns:
                if column not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _require_conn(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("ResultsRepository.connect() must be called first")
        return self.conn


def signal_identifier(signal: DiscoveredSignal) -> str:
    return deterministic_id(
        "signal",
        {
            "config_id": signal.config_id,
            "market_id": signal.market_id,
            "direction": signal.direction,
            "signal_ts": signal.signal_ts,
        },
    )


def trade_identifier(result: TradeResult) -> str:
    return deterministic_id(
        "trade",
        {
            "strategy_id": result.strategy_id,
            "signal_id": result.signal_id,
            "entry_actual_ts": result.entry_actual_ts,
            "exit_ts": result.exit_ts,
            "exit_reason": result.exit_reason,
        },
    )
