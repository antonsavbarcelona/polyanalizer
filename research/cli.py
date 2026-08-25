"""Command line interface for the offline research analyzer."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from research.config import ConfigError, load_experiment_config
from research.data.validator import validate_collector_db
from research.discovery_config import load_signal_discovery_config
from research.discovery_experiment import run_signal_discovery_experiment
from research.experiment import Experiment
from research.discovery_types import ALL_ASSET
from research.storage.discovery_repository import DiscoveryRepository
from research.storage.results_repository import ResultsRepository


DEFAULT_DATA = {
    "BTC": "data/analyzer_btc.db",
    "ETH": "data/analyzer_eth.db",
    "SOL": "data/analyzer_sol.db",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-data")
    validate.add_argument("db_paths", nargs="*")

    run = sub.add_parser("run-experiment")
    run.add_argument("config")

    top = sub.add_parser("top")
    top.add_argument("--experiment", required=True)
    top.add_argument("--results-db", default="data/research_results.db")
    top.add_argument("--limit", type=int, default=20)

    inspect_strategy = sub.add_parser("inspect-strategy")
    inspect_strategy.add_argument("strategy_id")
    inspect_strategy.add_argument("--experiment", required=True)
    inspect_strategy.add_argument("--results-db", default="data/research_results.db")

    inspect_trade = sub.add_parser("inspect-trade")
    inspect_trade.add_argument("trade_result_id")
    inspect_trade.add_argument("--experiment")
    inspect_trade.add_argument("--results-db", default="data/research_results.db")

    run_sr = sub.add_parser("run-signal-discovery")
    run_sr.add_argument("config")

    top_sr = sub.add_parser("top-signals")
    top_sr.add_argument("--experiment", required=True)
    top_sr.add_argument("--results-db", default="data/signal_discovery_results.db")
    top_sr.add_argument("--horizon-ms", type=int, default=2_000)
    top_sr.add_argument("--latency-ms", type=int, default=250)
    top_sr.add_argument("--size-shares", type=float, default=100.0)
    top_sr.add_argument("--limit", type=int, default=20)

    inspect_signal = sub.add_parser("inspect-signal")
    inspect_signal.add_argument("signal_id")
    inspect_signal.add_argument("--results-db", default="data/signal_discovery_results.db")
    inspect_signal.add_argument("--latency", type=int, default=250)
    inspect_signal.add_argument("--size", type=float, default=100.0)

    inspect_sc = sub.add_parser("inspect-signal-config")
    inspect_sc.add_argument("signal_config_id")
    inspect_sc.add_argument("--experiment", required=True)
    inspect_sc.add_argument("--results-db", default="data/signal_discovery_results.db")
    inspect_sc.add_argument("--latency", type=int, default=250)
    inspect_sc.add_argument("--size", type=float, default=100.0)

    compare_sc = sub.add_parser("compare-signal-configs")
    compare_sc.add_argument("config1")
    compare_sc.add_argument("config2")
    compare_sc.add_argument("--experiment", required=True)
    compare_sc.add_argument("--results-db", default="data/signal_discovery_results.db")
    compare_sc.add_argument("--latency", type=int, default=250)
    compare_sc.add_argument("--size", type=float, default=100.0)

    args = parser.parse_args(argv)
    try:
        if args.command == "validate-data":
            return _cmd_validate(args.db_paths)
        if args.command == "run-experiment":
            return _cmd_run(args.config)
        if args.command == "top":
            return _cmd_top(args.results_db, args.experiment, args.limit)
        if args.command == "inspect-strategy":
            return _cmd_inspect_strategy(args.results_db, args.experiment, args.strategy_id)
        if args.command == "inspect-trade":
            return _cmd_inspect_trade(args.results_db, args.trade_result_id, args.experiment)
        if args.command == "run-signal-discovery":
            return _cmd_run_signal_discovery(args.config)
        if args.command == "top-signals":
            return _cmd_top_signals(args.results_db, args.experiment, args.horizon_ms,
                                     args.latency_ms, args.size_shares, args.limit)
        if args.command == "inspect-signal":
            return _cmd_inspect_signal(args.results_db, args.signal_id, args.latency, args.size)
        if args.command == "inspect-signal-config":
            return _cmd_inspect_signal_config(args.results_db, args.experiment, args.signal_config_id,
                                               args.latency, args.size)
        if args.command == "compare-signal-configs":
            return _cmd_compare_signal_configs(args.results_db, args.experiment, args.config1, args.config2,
                                                args.latency, args.size)
    except (ConfigError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1


def _cmd_validate(db_paths: list[str]) -> int:
    assets = _assets_from_args(db_paths)
    reports = [validate_collector_db(asset, path) for asset, path in assets.items()]
    ok = True
    for report in reports:
        ok = ok and report.ok
        print(report.asset)
        print(f"  path: {report.db_path}")
        print(f"  markets: {report.markets}")
        for table, count in sorted(report.row_counts.items()):
            print(f"  {table}: {count}")
        print(f"  invalid_books: {report.invalid_books}")
        print(f"  missing_metadata: {report.missing_metadata}")
        if report.errors:
            for error in report.errors:
                print(f"  ERROR: {error}")
        print()
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED")
    return 0 if ok else 2


def _cmd_run(config_path: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_experiment_config(config_path)
    result = Experiment(config).run()
    print(f"experiment_id: {result.experiment_id}")
    print(f"status: SUCCESS")
    print(f"strategies: {result.strategies}")
    print(f"signals: {result.signals}")
    print(f"entries: {result.entries}")
    print(f"not_executable: {result.not_executable}")
    print(f"trades: {result.trades}")
    print()
    _print_validation_summary(result.validation_reports)
    print("leaderboard:")
    _print_leaderboard(config.results_db, result.experiment_id, limit=10)
    return 0


def _cmd_top(results_db: str, experiment_id: str, limit: int) -> int:
    _print_leaderboard(results_db, experiment_id, limit=limit)
    return 0


def _cmd_inspect_strategy(results_db: str, experiment_id: str, strategy_id: str) -> int:
    with ResultsRepository(results_db) as repo:
        bundle = repo.get_strategy_bundle(experiment_id, strategy_id)
    if bundle["strategy"] is None or bundle["metrics"] is None:
        raise RuntimeError(f"strategy not found: {strategy_id}")

    print(f"strategy_id: {strategy_id}")
    print("config:")
    print(_pretty_json(bundle["strategy"]["config_json"], indent=2))
    print("metrics:")
    print(_pretty_json(bundle["metrics"]["payload_json"], indent=2))
    print("per_asset:")
    for row in bundle["metrics_by_asset"]:
        payload = json.loads(row["payload_json"])
        print(f"  {row['asset']}: trades={payload['trade_count']} net_EV={_fmt(payload['mean_net_pnl'])} winrate={_fmt(payload['win_rate'])}")
    print("per_market:")
    for row in bundle["metrics_by_market"][:20]:
        payload = json.loads(row["payload_json"])
        print(f"  {row['market_id']}: trades={payload['trade_count']} net_EV={_fmt(payload['mean_net_pnl'])} winrate={_fmt(payload['win_rate'])}")
    print("bootstrap:")
    for row in bundle["bootstrap"]:
        print(f"  {row['metric']}: [{_fmt(row['low'])}, {_fmt(row['high'])}] @ {row['confidence']}")
    print("best_trades:")
    for row in bundle["best_trades"]:
        print(f"  {row['trade_result_id']} net={_fmt(row['net_pnl_per_share'])} reason={row['exit_reason']}")
    print("worst_trades:")
    for row in bundle["worst_trades"]:
        print(f"  {row['trade_result_id']} net={_fmt(row['net_pnl_per_share'])} reason={row['exit_reason']}")
    return 0


def _cmd_inspect_trade(results_db: str, trade_result_id: str, experiment_id: str | None) -> int:
    with ResultsRepository(results_db) as repo:
        trade = repo.get_trade(trade_result_id, experiment_id)
        if trade is None:
            raise RuntimeError(f"trade not found: {trade_result_id}")
        conn = repo.conn
        strategy = conn.execute("SELECT * FROM strategies WHERE strategy_id=?", (trade["strategy_id"],)).fetchone()
        signal = conn.execute(
            "SELECT * FROM signals WHERE experiment_id=? AND signal_id=?",
            (trade["experiment_id"], trade["signal_id"]),
        ).fetchone()
        entry = conn.execute(
            """
            SELECT e.* FROM entry_marks e
            JOIN strategies s ON s.execution_config_id=e.execution_config_id
            WHERE e.experiment_id=? AND e.signal_id=? AND s.strategy_id=?
            """,
            (trade["experiment_id"], trade["signal_id"], trade["strategy_id"]),
        ).fetchone()

    print(f"trade_result_id: {trade['trade_result_id']}")
    print(f"strategy_id: {trade['strategy_id']}")
    print(f"asset: {trade['asset']}")
    print(f"market_id: {trade['market_id']}")
    print(f"direction: {trade['direction']}")
    print(f"signal_ts: {trade['signal_ts']}")
    if signal is not None:
        print("signal_snapshot:")
        print(_pretty_json(signal["snapshot_json"], indent=2))
    if strategy is not None:
        print("strategy_config:")
        print(_pretty_json(strategy["config_json"], indent=2))
    if entry is not None:
        print("entry:")
        print(f"  requested_ts: {entry['requested_ts']}")
        print(f"  actual_ts: {entry['actual_ts']}")
        print(f"  best_ask: {_fmt(entry['best_ask'])}")
        print(f"  entry_vwap: {_fmt(entry['entry_vwap'])}")
        print(f"  entry_slippage: {_fmt(entry['entry_slippage'])}")
        print(f"  status: {entry['status']}")
    print("exit:")
    print(f"  exit_ts: {trade['exit_ts']}")
    print(f"  exit_price: {_fmt(trade['exit_price'])}")
    print(f"  exit_reason: {trade['exit_reason']}")
    print("economics:")
    print(f"  gross_pnl_per_share: {_fmt(trade['gross_pnl_per_share'])}")
    print(f"  entry_fee_per_share: {_fmt(trade['entry_fee_per_share'])}")
    print(f"  exit_fee_per_share: {_fmt(trade['exit_fee_per_share'])}")
    print(f"  total_fee_per_share: {_fmt(trade['fees_per_share'])}")
    print(f"  net_pnl_per_share: {_fmt(trade['net_pnl_per_share'])}")
    print(f"  pnl_total: {_fmt(trade['pnl_total'])}")
    print(f"  mfe: {_fmt(trade['mfe'])}")
    print(f"  mae: {_fmt(trade['mae'])}")
    return 0


def _print_validation_summary(reports) -> None:
    print("validation:")
    for report in reports:
        print(
            f"  {report.asset}: markets={report.markets} "
            f"trades={report.row_counts.get('binance_trades', 0)} "
            f"books={report.row_counts.get('binance_book', 0)} "
            f"poly={report.row_counts.get('polymarket_book', 0)}"
        )
    print()


def _print_leaderboard(results_db: str, experiment_id: str, limit: int) -> None:
    with ResultsRepository(results_db) as repo:
        rows = repo.leaderboard(experiment_id, limit=limit)
    if not rows:
        print("  empty")
        return
    print(
        "  strategy_id                 signals entries trades markets winrate net_EV gross_EV profit_factor CI_net"
    )
    for row in rows:
        payload = json.loads(row["payload_json"])
        markets = _markets_for_strategy(results_db, experiment_id, row["strategy_id"])
        print(
            f"  {row['strategy_id']:<27} "
            f"{payload['signal_count']:>7} {payload['entry_count']:>7} {payload['trade_count']:>6} "
            f"{markets:>7} {_fmt(payload['win_rate']):>7} {_fmt(payload['mean_net_pnl']):>7} "
            f"{_fmt(payload['mean_gross_pnl']):>8} {_fmt(payload['profit_factor']):>13} "
            f"[{_fmt(row['net_ci_low'])},{_fmt(row['net_ci_high'])}]"
        )


def _markets_for_strategy(results_db: str, experiment_id: str, strategy_id: str) -> int:
    with ResultsRepository(results_db) as repo:
        return repo.conn.execute(
            """
            SELECT COUNT(*) FROM strategy_metrics_by_market
            WHERE experiment_id=? AND strategy_id=?
            """,
            (experiment_id, strategy_id),
        ).fetchone()[0]


def _assets_from_args(db_paths: list[str]) -> dict[str, str]:
    if not db_paths:
        return DEFAULT_DATA
    out = {}
    for path in db_paths:
        stem = Path(path).stem.lower()
        if "btc" in stem:
            asset = "BTC"
        elif "eth" in stem:
            asset = "ETH"
        elif "sol" in stem:
            asset = "SOL"
        else:
            asset = stem.upper()
        out[asset] = path
    return out


def _pretty_json(value: str, indent: int = 2) -> str:
    return json.dumps(json.loads(value), indent=indent, sort_keys=True)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.6f}"
    return str(value)


# ---------------------------------------------------------------------------
# Signal-response discovery pipeline commands (IMPLEMENTATION CONTRACT #2)
# ---------------------------------------------------------------------------

def _cmd_run_signal_discovery(config_path: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_signal_discovery_config(config_path)
    result = run_signal_discovery_experiment(config)
    print(f"experiment_id: {result.experiment_id}")
    print("status: SUCCESS")
    print(f"signal_configs: {result.signal_configs}")
    print(f"signals: {result.signals}")
    print(f"executable_entries: {result.executable_entries}")
    print(f"controls: {result.controls}")
    print()
    print("timing breakdown (seconds, summed across worker processes):")
    for key in sorted(result.timings):
        print(f"  {key}: {result.timings[key]:.2f}")
    return 0


def _cmd_top_signals(results_db: str, experiment_id: str, horizon_ms: int,
                      latency_ms: int, size_shares: float, limit: int) -> int:
    with DiscoveryRepository(results_db) as repo:
        rows = repo.top_signal_configs(experiment_id, horizon_ms, latency_ms, size_shares, limit=limit)
    if not rows:
        print("  empty")
        return 0
    print(f"  canonical slice: asset=ALL latency={latency_ms}ms size={size_shares} horizon={horizon_ms}ms")
    print(
        "  signal_config_id            signals markets exec_rate avail  p_pos  mean   uplift_mean plateau_score"
    )
    for row in rows:
        print(
            f"  {row['signal_config_id']:<27} "
            f"{row['signal_count']:>7} {row['market_count']:>7} {_fmt(row['entry_execution_rate']):>9} "
            f"{row['available_count']:>5} {_fmt(row['p_positive']):>6} {_fmt(row['mean_response']):>7} "
            f"{_fmt(row['uplift_mean_response']):>11} {_fmt(row['plateau_score']):>13}"
        )
    return 0


def _cmd_inspect_signal(results_db: str, signal_id: str, latency_ms: int, size_shares: float) -> int:
    with DiscoveryRepository(results_db) as repo:
        conn = repo.conn
        signal = conn.execute("SELECT * FROM signals WHERE signal_id=?", (signal_id,)).fetchone()
        if signal is None:
            raise RuntimeError(f"signal not found: {signal_id}")
        entry = conn.execute(
            "SELECT * FROM entry_marks WHERE signal_id=? AND latency_ms=? AND size_shares=?",
            (signal_id, latency_ms, size_shares),
        ).fetchone()
        responses = conn.execute(
            "SELECT * FROM signal_response WHERE signal_id=? AND latency_ms=? AND size_shares=? ORDER BY horizon_ms",
            (signal_id, latency_ms, size_shares),
        ).fetchall()
        path_stats = None
        if entry is not None:
            path_stats = conn.execute(
                "SELECT * FROM signal_path_stats WHERE entry_mark_id=? ORDER BY stats_horizon_ms DESC LIMIT 1",
                (entry["entry_mark_id"],),
            ).fetchone()
        controls = conn.execute(
            "SELECT * FROM controls WHERE source_signal_id=? ORDER BY match_rank", (signal_id,),
        ).fetchall()

    snapshot = json.loads(signal["snapshot_json"])
    print(f"SIGNAL {signal_id}")
    print(f"  signal_config_id: {signal['signal_config_id']}")
    print(f"  asset: {signal['asset']}  market: {signal['market_id']}  direction: {signal['direction']}")
    print(f"  signal_ts: {signal['signal_ts']}")
    print("FEATURES")
    for key in ("z_score", "active_momentum_value", "active_flow", "imbalance_top5", "poly_move_500ms",
                "tte_regime", "price_regime", "spread_regime", "volatility_regime"):
        print(f"  {key}: {snapshot.get(key)}")

    print(f"\nENTRY (latency={latency_ms}ms size={size_shares})")
    if entry is None:
        print("  no entry_mark computed for this latency/size")
    else:
        print(f"  status: {entry['status']}")
        print(f"  target_ts: {entry['entry_target_ts']}  actual_ts: {entry['entry_actual_ts']}")
        print(f"  best_ask: {_fmt(entry['entry_best_ask'])}  entry_vwap: {_fmt(entry['entry_vwap'])}")
        print(f"  entry_slippage: {_fmt(entry['entry_slippage'])}")
        print(f"  entry_fee_per_share: {_fmt(entry['entry_fee_per_share'])}")

    print("\nRESPONSES")
    for r in responses:
        print(
            f"  {r['horizon_ms']:>6}ms: status={r['status']:<20} sell_vwap={_fmt(r['future_sell_vwap']):>10} "
            f"raw={_fmt(r['raw_response']):>10} fee_adj={_fmt(r['fee_adjusted_response']):>10}"
        )

    if path_stats is not None:
        print(f"\nPATH STATS (stats_horizon={path_stats['stats_horizon_ms']}ms)")
        print(f"  mfe: {_fmt(path_stats['mfe'])}  time_to_mfe_ms: {path_stats['time_to_mfe_ms']}")
        print(f"  mae: {_fmt(path_stats['mae'])}  time_to_mae_ms: {path_stats['time_to_mae_ms']}")
        print(f"  time_to_plus_010_ms: {path_stats['time_to_plus_010_ms']}")
        print(f"  time_to_minus_010_ms: {path_stats['time_to_minus_010_ms']}")

    print(f"\nMATCHED CONTROLS ({len(controls)})")
    for c in controls:
        print(f"  rank={c['match_rank']} ts={c['control_ts']} distance={_fmt(c['match_distance'])}")
    return 0


def _cmd_inspect_signal_config(results_db: str, experiment_id: str, signal_config_id: str,
                                latency_ms: int, size_shares: float) -> int:
    with DiscoveryRepository(results_db) as repo:
        conn = repo.conn
        cfg = conn.execute("SELECT * FROM signal_configs WHERE signal_config_id=?", (signal_config_id,)).fetchone()
        if cfg is None:
            raise RuntimeError(f"signal_config not found: {signal_config_id}")
        overall = conn.execute(
            """SELECT * FROM signal_config_summary
               WHERE experiment_id=? AND signal_config_id=? AND asset=?
                 AND latency_ms=? AND size_shares=?""",
            (experiment_id, signal_config_id, ALL_ASSET, latency_ms, size_shares),
        ).fetchone()
        by_asset = conn.execute(
            """SELECT * FROM signal_config_summary
               WHERE experiment_id=? AND signal_config_id=? AND asset<>?
                 AND latency_ms=? AND size_shares=?""",
            (experiment_id, signal_config_id, ALL_ASSET, latency_ms, size_shares),
        ).fetchall()
        responses = conn.execute(
            """SELECT * FROM signal_response_summary
               WHERE experiment_id=? AND signal_config_id=? AND asset=?
                 AND latency_ms=? AND size_shares=? ORDER BY horizon_ms""",
            (experiment_id, signal_config_id, ALL_ASSET, latency_ms, size_shares),
        ).fetchall()

    print(f"CONFIG {signal_config_id}")
    print(_pretty_json(cfg["config_json"], indent=2))
    if overall is not None:
        print(f"\nsignals: {overall['signal_count']}  markets: {overall['market_count']}  "
              f"plateau_score: {_fmt(overall['plateau_score'])}")
    print("\nby asset:")
    for row in by_asset:
        print(f"  {row['asset']}: signals={row['signal_count']} markets={row['market_count']} "
              f"exec_rate={_fmt(row['entry_execution_rate'])}")
    print(f"\nresponses (latency={latency_ms}ms size={size_shares}):")
    print("  horizon_ms  avail  p_pos   mean    control_mean  uplift_mean")
    for row in responses:
        print(
            f"  {row['horizon_ms']:>9}  {row['available_count']:>5}  {_fmt(row['p_positive']):>6}  "
            f"{_fmt(row['mean_response']):>6}  {_fmt(row['control_mean_response']):>12}  "
            f"{_fmt(row['uplift_mean_response']):>11}"
        )
    return 0


def _cmd_compare_signal_configs(results_db: str, experiment_id: str, config1: str, config2: str,
                                 latency_ms: int, size_shares: float) -> int:
    with DiscoveryRepository(results_db) as repo:
        conn = repo.conn
        rows = {
            cid: conn.execute(
                """SELECT * FROM signal_response_summary
                   WHERE experiment_id=? AND signal_config_id=? AND asset=?
                     AND latency_ms=? AND size_shares=? ORDER BY horizon_ms""",
                (experiment_id, cid, ALL_ASSET, latency_ms, size_shares),
            ).fetchall()
            for cid in (config1, config2)
        }
    print(f"COMPARE {config1} vs {config2}  (latency={latency_ms}ms size={size_shares})")
    print(f"  {'horizon_ms':>10}  {'mean('+config1[:10]+')':>16}  {'mean('+config2[:10]+')':>16}")
    by_horizon = {config1: {r["horizon_ms"]: r for r in rows[config1]},
                  config2: {r["horizon_ms"]: r for r in rows[config2]}}
    horizons = sorted(set(by_horizon[config1]) | set(by_horizon[config2]))
    for h in horizons:
        r1 = by_horizon[config1].get(h)
        r2 = by_horizon[config2].get(h)
        print(f"  {h:>10}  {_fmt(r1['mean_response'] if r1 else None):>16}  {_fmt(r2['mean_response'] if r2 else None):>16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
