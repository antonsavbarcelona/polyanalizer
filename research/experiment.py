"""End-to-end experiment orchestration."""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from poly_analyzer.discovery import (
    DiscoveredSignal,
    DiscoverySignalConfig,
    ExecutionMark,
    list_market_ids,
    load_market_meta,
    replay_signals,
)

from research.analytics.bootstrap import bootstrap_confidence_intervals
from research.analytics.metrics import compute_strategy_metrics
from research.config import ExperimentConfig
from research.data.validator import ValidationReport, fingerprint_db, open_readonly, validate_collector_db
from research.execution.episode import build_episode_from_entry, compute_research_entry_mark
from research.execution.exit_policy import evaluate_exit
from research.fees import FeeModel
from research.grid import generate_execution_configs, generate_exit_configs, generate_signal_configs
from research.storage.results_repository import ResultsRepository, signal_identifier
from research.types import (
    BootstrapResult,
    ExecutionConfig,
    ExitConfig,
    SignalConfig,
    StrategyConfig,
    StrategyMetrics,
    TradeResult,
)

log = logging.getLogger(__name__)

MAX_WORKERS = 4


@dataclass
class ExperimentRunResult:
    experiment_id: str
    validation_reports: list[ValidationReport]
    signals: int
    entries: int
    not_executable: int
    trades: int
    strategies: int


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.path_build_count = 0
        self.signal_replay_count = 0

    def run(self) -> ExperimentRunResult:
        validation_reports = [validate_collector_db(asset, path) for asset, path in self.config.data.items()]
        failed = [report for report in validation_reports if not report.ok]
        if failed:
            details = "; ".join(f"{report.asset}: {', '.join(report.errors)}" for report in failed)
            raise RuntimeError(f"data validation failed: {details}")

        data_fingerprint = {
            report.asset: {
                "path": report.db_path,
                "fingerprint": report.fingerprint,
                "rows": report.row_counts,
                "markets": report.markets,
            }
            for report in validation_reports
        }
        signal_configs = generate_signal_configs(self.config)
        execution_configs = generate_execution_configs(self.config)
        exit_configs = generate_exit_configs(self.config)
        strategies = [
            StrategyConfig(signal=signal, execution=execution, exit=exit_cfg)
            for signal in signal_configs
            for execution in execution_configs
            for exit_cfg in exit_configs
        ]
        if not strategies:
            raise RuntimeError("strategy grid is empty")

        repo = ResultsRepository(self.config.results_db)
        repo.connect()
        experiment_id = repo.create_experiment(
            self.config.name,
            self.config,
            data_sources=self.config.data,
            data_fingerprint=json.dumps(data_fingerprint, sort_keys=True),
            code_version=_code_version(),
            fee_model=self.config.fee_model,
            research_clock_ms=self.config.research_clock_ms,
        )

        signals_total = 0
        entries_total = 0
        not_executable_total = 0
        trades_total = 0
        strategies_checkpointed = 0

        try:
            for strategy in strategies:
                repo.save_strategy_config(strategy)

            strategies_by_signal: dict[str, list[StrategyConfig]] = defaultdict(list)
            for strategy in strategies:
                strategies_by_signal[strategy.signal.id].append(strategy)

            # Each signal_config's replay+evaluation is fully independent of
            # every other signal_config's (no shared mutable state -- it's
            # pure re-computation over already-collected, immutable raw
            # data), so it's dispatched to its own worker process. The
            # SQLite results DB has a single writer: workers return plain,
            # picklable result bundles and this process does every repo.save_*
            # call itself, as each signal_config finishes -- which keeps the
            # same "checkpoint as soon as a signal_config is fully done"
            # property as a sequential run, just arriving out of order.
            #
            # MAX_WORKERS is capped, not os.cpu_count(): measured on this
            # machine (12 physical / 24 logical cores), 16 concurrent worker
            # processes never even finished spawning (stuck 30+ minutes,
            # zero children), while 4 spawned and ran reliably -- some
            # environment-specific ceiling (likely AV/EDR or a sandbox
            # process-count limit) sits well below the core count. 4 is the
            # largest value confirmed to work end-to-end on real data.
            max_workers = min(len(signal_configs), MAX_WORKERS) or 1
            log.info(
                "replaying %d signal_configs across %d worker processes",
                len(signal_configs), max_workers,
            )
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _run_signal_config,
                        signal_cfg,
                        self.config.data,
                        self.config.max_markets_per_asset,
                        execution_configs,
                        strategies_by_signal[signal_cfg.id],
                        self.config.bootstrap_iterations,
                    ): signal_cfg
                    for signal_cfg in signal_configs
                }
                completed = 0
                for future in as_completed(futures):
                    signal_cfg = futures[future]
                    result = future.result()
                    completed += 1

                    for signal, asset in result.signals:
                        repo.save_signal(experiment_id, signal, asset)
                    for signal_id, execution_config_id, mark, status in result.entry_marks:
                        repo.save_entry_mark(experiment_id, signal_id, execution_config_id, mark, status)
                    for trade in result.trades:
                        repo.save_trade_result(trade, experiment_id)
                    for metrics in result.metrics:
                        repo.save_strategy_metrics(metrics, experiment_id)
                    for bootstrap in result.bootstrap:
                        repo.save_bootstrap_result(bootstrap, experiment_id)

                    signals_total += result.signals_total
                    entries_total += result.entries_total
                    not_executable_total += result.not_executable_total
                    trades_total += result.trades_total
                    self.signal_replay_count += result.signal_replay_count
                    self.path_build_count += result.path_build_count
                    strategies_checkpointed += len(strategies_by_signal[signal_cfg.id])

                    log.info(
                        "checkpointed signal_config %d/%d (%s): %d/%d strategies, %d signals, %d trades so far",
                        completed, len(signal_configs), signal_cfg.id,
                        strategies_checkpointed, len(strategies), signals_total, trades_total,
                    )

            after = {asset: fingerprint_db(path) for asset, path in self.config.data.items()}
            before = {report.asset: report.fingerprint for report in validation_reports}
            if before != after:
                raise RuntimeError("collector DB fingerprint changed during research run")

            repo.finish_experiment(experiment_id, "SUCCESS")
        except Exception as exc:
            repo.finish_experiment(experiment_id, "FAILED", {"error": str(exc)})
            repo.close()
            raise

        repo.close()
        return ExperimentRunResult(
            experiment_id=experiment_id,
            validation_reports=validation_reports,
            signals=signals_total,
            entries=entries_total,
            not_executable=not_executable_total,
            trades=trades_total,
            strategies=len(strategies),
        )


def to_discovery_config(config: SignalConfig) -> DiscoverySignalConfig:
    return DiscoverySignalConfig(
        momentum_window_ms=config.momentum_window_ms,
        volatility_lookback_ms=config.volatility_window_ms,
        z_threshold=config.z_threshold,
        flow_window_ms=config.flow_window_ms or config.momentum_window_ms,
        flow_threshold=config.flow_threshold if config.flow_threshold is not None else -1.0,
        imbalance_depth=config.imbalance_depth or 5,
        imbalance_threshold=config.imbalance_threshold if config.imbalance_threshold is not None else -1.0,
        poly_lag_window_ms=config.poly_lag_window_ms or 500,
        max_poly_reprice=config.max_poly_reprice if config.max_poly_reprice is not None else 1.0,
        ask_min=config.min_contract_price if config.min_contract_price is not None else 0.0,
        ask_max=config.max_contract_price if config.max_contract_price is not None else 1.0,
        spread_max=config.max_spread if config.max_spread is not None else 1.0,
        tte_min_s=(config.min_time_remaining_ms / 1000.0) if config.min_time_remaining_ms is not None else 0.0,
        tte_max_s=(config.max_time_remaining_ms / 1000.0) if config.max_time_remaining_ms is not None else 10**9,
        reset_z=config.reset_z,
        cooldown_ms=config.cooldown_ms,
        sigma_min_samples=5,
    )


def _market_row_with_defaults(conn: sqlite3.Connection, market_id: str, asset: str) -> dict[str, Any]:
    row = load_market_meta(conn, market_id) or {"market_id": market_id}
    settlement = conn.execute("SELECT * FROM market_settlement WHERE market_id=?", (market_id,)).fetchone()
    if settlement is not None:
        row.update(dict(settlement))
    row.setdefault("asset", asset)
    if row.get("fee_rate") is None:
        row["fee_rate"] = 0.07
    if row.get("fee_exponent") is None:
        row["fee_exponent"] = 1.0
    return row


def _increment(
    counters: dict[tuple[str, str | None, str | None], dict[str, int]],
    strategy_id: str,
    asset: str,
    market_id: str,
    key: str,
) -> None:
    counters[(strategy_id, None, None)][key] += 1
    counters[(strategy_id, asset, None)][key] += 1
    counters[(strategy_id, None, market_id)][key] += 1


@dataclass
class SignalConfigResult:
    """Everything one signal_config's worker computed, ready for the parent
    process to write via ResultsRepository (the single DB writer). All
    fields are plain dataclasses/tuples so this crosses the process
    boundary via pickling with no extra work."""

    signal_config_id: str
    signals: list[tuple[DiscoveredSignal, str]] = field(default_factory=list)  # (signal, asset)
    entry_marks: list[tuple[str, str, ExecutionMark, str]] = field(default_factory=list)  # (signal_id, exec_cfg_id, mark, status)
    trades: list[TradeResult] = field(default_factory=list)
    metrics: list[StrategyMetrics] = field(default_factory=list)
    bootstrap: list[BootstrapResult] = field(default_factory=list)
    signals_total: int = 0
    entries_total: int = 0
    not_executable_total: int = 0
    trades_total: int = 0
    signal_replay_count: int = 0
    path_build_count: int = 0


def _run_signal_config(
    signal_cfg: SignalConfig,
    data: dict[str, str],
    max_markets_per_asset: int | None,
    execution_configs: list[ExecutionConfig],
    strategies: list[StrategyConfig],
    bootstrap_iterations: int,
) -> SignalConfigResult:
    """Worker entry point (runs in its own process): replays one
    signal_config across every asset/market, evaluates every (execution,
    exit) strategy built on it, and computes that strategy's metrics +
    bootstrap CIs -- entirely from data it reads itself (each worker opens
    its own read-only SQLite connections; nothing is shared across the
    process boundary). Pure computation over immutable, already-collected
    raw data, so signal_configs are fully independent of each other and
    this is safe to run in parallel with no coordination beyond the results
    each worker hands back."""
    discovery_cfg = to_discovery_config(signal_cfg)
    strategies_by_exec: dict[str, list[tuple[ExitConfig, str]]] = defaultdict(list)
    for strategy in strategies:
        strategies_by_exec[strategy.execution.id].append((strategy.exit, strategy.strategy_id()))

    trades_by_strategy: dict[str, list[TradeResult]] = defaultdict(list)
    counters: dict[tuple[str, str | None, str | None], dict[str, int]] = defaultdict(
        lambda: {"signals": 0, "entries": 0, "not_executable": 0}
    )
    result = SignalConfigResult(signal_config_id=signal_cfg.id)

    for asset, db_path in data.items():
        conn = open_readonly(db_path)
        try:
            market_ids = list_market_ids(conn)
            if max_markets_per_asset is not None:
                market_ids = market_ids[:max_markets_per_asset]
            for market_id in market_ids:
                result.signal_replay_count += 1
                signals = replay_signals(conn, market_id, discovery_cfg, config_id=signal_cfg.id)
                if not signals:
                    continue
                market_row = _market_row_with_defaults(conn, market_id, asset)

                for signal in signals:
                    result.signals_total += 1
                    signal_id = signal_identifier(signal)
                    result.signals.append((signal, asset))

                    for execution_cfg in execution_configs:
                        mark, status = compute_research_entry_mark(
                            conn,
                            signal.market_id,
                            signal.direction,
                            signal.signal_ts,
                            execution_cfg.latency_ms,
                            execution_cfg.size_shares,
                            market_row.get("end_ts"),
                        )
                        result.entry_marks.append((signal_id, execution_cfg.id, mark, status))
                        episode = None
                        if status == "EXECUTED":
                            result.entries_total += 1
                            episode = build_episode_from_entry(conn, signal, execution_cfg.id, mark, market_row)
                            result.path_build_count += 1
                        else:
                            result.not_executable_total += 1

                        for exit_cfg, strategy_id in strategies_by_exec[execution_cfg.id]:
                            _increment(counters, strategy_id, asset, signal.market_id, "signals")
                            if status != "EXECUTED":
                                _increment(counters, strategy_id, asset, signal.market_id, "not_executable")
                                continue
                            _increment(counters, strategy_id, asset, signal.market_id, "entries")
                            trade = evaluate_exit(episode, exit_cfg, FeeModel(), execution_cfg.size_shares, asset=asset)
                            if trade is None:
                                continue
                            trade.strategy_id = strategy_id
                            trade.trade_result_id = ""
                            result.trades.append(trade)
                            trades_by_strategy[strategy_id].append(trade)
                            result.trades_total += 1
        finally:
            conn.close()

    for strategy in strategies:
        strategy_id = strategy.strategy_id()
        trades = trades_by_strategy[strategy_id]

        all_key = (strategy_id, None, None)
        result.metrics.append(compute_strategy_metrics(
            strategy_id,
            trades,
            signal_count=counters[all_key]["signals"],
            entry_count=counters[all_key]["entries"],
            not_executable_count=counters[all_key]["not_executable"],
        ))

        for asset in sorted(data):
            key = (strategy_id, asset, None)
            asset_trades = [trade for trade in trades if trade.asset == asset]
            result.metrics.append(compute_strategy_metrics(
                strategy_id,
                asset_trades,
                signal_count=counters[key]["signals"],
                entry_count=counters[key]["entries"],
                not_executable_count=counters[key]["not_executable"],
                asset=asset,
            ))

        market_keys = sorted(key for key in counters if key[0] == strategy_id and key[2] is not None)
        for _, _asset, market_id in market_keys:
            key = (strategy_id, None, market_id)
            market_trades = [trade for trade in trades if trade.market_id == market_id]
            result.metrics.append(compute_strategy_metrics(
                strategy_id,
                market_trades,
                signal_count=counters[key]["signals"],
                entry_count=counters[key]["entries"],
                not_executable_count=counters[key]["not_executable"],
                market_id=market_id,
            ))

        result.bootstrap.extend(
            bootstrap_confidence_intervals(trades, strategy_id, iterations=bootstrap_iterations)
        )

    return result


def _code_version() -> str:
    return "unknown"
