import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from tests.discovery_fixtures import (
    binance_book_row,
    binance_trade_row,
    insert_rows,
    make_recorder,
    market_row,
    poly_book_row,
)

ROOT = Path(__file__).resolve().parents[2]


def _run_cli(args, *, cwd=ROOT):
    env = os.environ.copy()
    env["TEMP"] = str(ROOT / "run_temp")
    env["TMP"] = env["TEMP"]
    return subprocess.run(
        [sys.executable, "-m", "research.cli", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _chainlink_row(market_id, ts=0):
    return {
        "market_id": market_id,
        "observation_ts": ts,
        "recv_ts": ts,
        "recv_monotonic_ns": ts * 1_000_000,
        "twap_value": 100.0,
        "twap_window": 60,
        "is_debug": 0,
    }


def _build_signal_market(rec, market_id, *, end_ts=10_000, entry_liquidity=100.0, tp_bid=0.56):
    insert_rows(rec, "markets", [market_row(market_id, end_ts=end_ts)])
    insert_rows(rec, "chainlink_observations", [_chainlink_row(market_id)])
    insert_rows(
        rec,
        "binance_book",
        [binance_book_row(market_id, ts, [(100.0, 50.0)], [(100.02, 50.0)]) for ts in range(0, 1_000, 100)]
        + [binance_book_row(market_id, 1_000, [(104.9, 400.0), (104.8, 400.0)], [(105.0, 20.0)])],
    )
    insert_rows(
        rec,
        "binance_trades",
        [
            binance_trade_row(market_id, 1, 900, 100.0, 1.0, "SELL"),
            binance_trade_row(market_id, 2, 950, 104.9, 5.0, "BUY"),
            binance_trade_row(market_id, 3, 990, 105.0, 5.0, "BUY"),
        ],
    )
    insert_rows(
        rec,
        "polymarket_book",
        [
            poly_book_row(market_id, 400, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)]),
            poly_book_row(market_id, 1_000, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)]),
            poly_book_row(market_id, 1_240, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)]),
            poly_book_row(
                market_id,
                1_260,
                up_bid=0.49,
                up_ask=0.50,
                up_bids=[(0.49, 100)],
                up_asks=[(0.50, 20), (0.51, 30), (0.52, entry_liquidity)],
            ),
            poly_book_row(market_id, 1_800, up_bid=tp_bid, up_ask=0.58, up_bids=[(tp_bid, 100)], up_asks=[(0.58, 100)]),
        ],
    )


def _build_fixture_db(tmp_path):
    rec = make_recorder(tmp_path)
    _build_signal_market(rec, "M_TP", entry_liquidity=100.0, tp_bid=0.56)
    _build_signal_market(rec, "M_NO_LIQ", entry_liquidity=20.0, tp_bid=0.56)
    rec.close()
    return tmp_path / "test.db"


def _write_config(
    tmp_path,
    db_path,
    result_db,
    *,
    name="config.yaml",
    signal_thresholds=(2.0,),
    latencies=(250,),
    tps=(0.04,),
    sls=(0.03,),
):
    config_path = tmp_path / name
    config_path.write_text(
        f"""
experiment:
  name: e2e
data:
  BTC: {db_path}
results_db: {result_db}
research_clock_ms: 100
bootstrap_iterations: 20
signal_grid:
  momentum_window_ms: [100]
  volatility_window_ms: [1000]
  z_threshold: {_yaml_list(signal_thresholds)}
  flow_window_ms: [500]
  flow_threshold: [0.3]
  max_spread: [0.05]
  min_contract_price: [0.05]
  max_contract_price: [0.95]
  min_time_remaining_ms: [0]
  max_time_remaining_ms: [1000000000]
execution_grid:
  latency_ms: {_yaml_list(latencies)}
  size_shares: [100]
exit_grid:
  tp_delta: {_yaml_list(tps)}
  sl_delta: {_yaml_list(sls)}
  max_holding_ms: [10000]
fee_model:
  name: fixed_v1
ranking:
  min_signals: 0
  min_markets: 0
""",
        encoding="utf-8",
    )
    return config_path


def _yaml_list(values):
    out = []
    for value in values:
        out.append("null" if value is None else str(value))
    return "[" + ", ".join(out) + "]"


def _experiment_id(stdout):
    for line in stdout.splitlines():
        if line.startswith("experiment_id:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(stdout)


def test_e2e_full_cli_happy_path_persists_results(tmp_path):
    db_path = _build_fixture_db(tmp_path)
    result_db = tmp_path / "results.db"
    config_path = _write_config(
        tmp_path,
        db_path,
        result_db,
        signal_thresholds=(2.0, 3.0),
        latencies=(100, 250),
        tps=(0.04, 0.05),
        sls=(None, 0.03),
    )

    proc = _run_cli(["run-experiment", str(config_path)])

    assert proc.returncode == 0, proc.stderr + proc.stdout
    experiment_id = _experiment_id(proc.stdout)
    conn = sqlite3.connect(result_db)
    try:
        assert conn.execute("SELECT status FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()[0] == "SUCCESS"
        assert conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0] == 16
        assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM entry_marks").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM trade_results").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM strategy_metrics").fetchone()[0] == 16
        assert conn.execute("SELECT COUNT(*) FROM bootstrap_results").fetchone()[0] > 0
    finally:
        conn.close()


def test_e2e_repeatability_ids_and_economics_are_stable(tmp_path):
    db_path = _build_fixture_db(tmp_path)
    config_a = _write_config(tmp_path, db_path, tmp_path / "a.db", name="a.yaml")
    config_b = _write_config(tmp_path, db_path, tmp_path / "b.db", name="b.yaml")

    run_a = _run_cli(["run-experiment", str(config_a)])
    run_b = _run_cli(["run-experiment", str(config_b)])

    assert run_a.returncode == 0, run_a.stderr
    assert run_b.returncode == 0, run_b.stderr

    def rows(path):
        conn = sqlite3.connect(path)
        try:
            return conn.execute(
                """
                SELECT strategy_id, signal_id, entry_vwap, exit_price, exit_reason,
                       gross_pnl_per_share, fees_per_share, net_pnl_per_share
                FROM trade_results
                ORDER BY strategy_id, signal_id, entry_actual_ts
                """
            ).fetchall()
        finally:
            conn.close()

    assert rows(tmp_path / "a.db") == rows(tmp_path / "b.db")


def test_e2e_latency_and_multilevel_vwap_are_used(tmp_path):
    db_path = _build_fixture_db(tmp_path)
    result_db = tmp_path / "results.db"
    config_path = _write_config(tmp_path, db_path, result_db)

    proc = _run_cli(["run-experiment", str(config_path)])

    assert proc.returncode == 0, proc.stderr + proc.stdout
    conn = sqlite3.connect(result_db)
    try:
        row = conn.execute(
            """
            SELECT entry_requested_ts, entry_actual_ts, entry_vwap, exit_reason, exit_price
            FROM trade_results
            WHERE market_id='M_TP'
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 1_250
    assert row[1] == 1_260
    assert row[2] == (20 * 0.50 + 30 * 0.51 + 50 * 0.52) / 100
    assert row[3] == "TP"
    assert row[4] == row[2] + 0.04


def test_e2e_insufficient_liquidity_rejects_entry(tmp_path):
    db_path = _build_fixture_db(tmp_path)
    result_db = tmp_path / "results.db"
    config_path = _write_config(tmp_path, db_path, result_db)

    proc = _run_cli(["run-experiment", str(config_path)])

    assert proc.returncode == 0, proc.stderr + proc.stdout
    conn = sqlite3.connect(result_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM trade_results WHERE market_id='M_NO_LIQ'").fetchone()[0] == 0
        status = conn.execute(
            """
            SELECT status FROM entry_marks e
            JOIN signals s ON s.signal_id=e.signal_id AND s.experiment_id=e.experiment_id
            WHERE s.market_id='M_NO_LIQ'
            """
        ).fetchone()[0]
        metrics = json.loads(conn.execute("SELECT payload_json FROM strategy_metrics").fetchone()[0])
    finally:
        conn.close()
    assert status == "NOT_EXECUTABLE"
    assert metrics["not_executable_count"] >= 1


def test_e2e_top_inspect_and_source_db_immutable(tmp_path):
    db_path = _build_fixture_db(tmp_path)
    before = _source_counts(db_path)
    result_db = tmp_path / "results.db"
    config_path = _write_config(tmp_path, db_path, result_db)

    run = _run_cli(["run-experiment", str(config_path)])
    experiment_id = _experiment_id(run.stdout)
    assert run.returncode == 0, run.stderr
    assert before == _source_counts(db_path)

    top = _run_cli(["top", "--experiment", experiment_id, "--results-db", str(result_db)])
    assert top.returncode == 0, top.stderr
    assert "strategy_" in top.stdout

    conn = sqlite3.connect(result_db)
    strategy_id, trade_id = conn.execute(
        "SELECT strategy_id, trade_result_id FROM trade_results LIMIT 1"
    ).fetchone()
    conn.close()

    inspect_strategy = _run_cli(
        ["inspect-strategy", strategy_id, "--experiment", experiment_id, "--results-db", str(result_db)]
    )
    inspect_trade = _run_cli(
        ["inspect-trade", trade_id, "--experiment", experiment_id, "--results-db", str(result_db)]
    )
    assert inspect_strategy.returncode == 0, inspect_strategy.stderr
    assert "metrics:" in inspect_strategy.stdout
    assert inspect_trade.returncode == 0, inspect_trade.stderr
    assert "economics:" in inspect_trade.stdout


def test_e2e_invalid_flow_combo_fails_before_experiment(tmp_path):
    db_path = _build_fixture_db(tmp_path)
    result_db = tmp_path / "results.db"
    config_path = _write_config(tmp_path, db_path, result_db)
    text = config_path.read_text(encoding="utf-8").replace("flow_threshold: [0.3]", "flow_threshold: [0.3]")
    text = text.replace("flow_window_ms: [500]", "flow_window_ms: [null]")
    config_path.write_text(text, encoding="utf-8")

    proc = _run_cli(["run-experiment", str(config_path)])

    assert proc.returncode != 0
    assert "signal_grid produced no valid SignalConfig" in proc.stderr


def _source_counts(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ["markets", "binance_trades", "binance_book", "polymarket_book", "chainlink_observations"]
        }
    finally:
        conn.close()
