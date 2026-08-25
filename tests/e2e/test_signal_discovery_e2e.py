"""End-to-end run of the signal-response discovery pipeline against a small
synthetic fixture with a real triggering Binance momentum spike -- exercises
detect_signals -> entry -> response -> path_stats -> controls -> summaries
-> plateau, all wired together through run_signal_discovery_experiment,
plus the CLI commands on top of the resulting DB."""
from __future__ import annotations

import json
import os
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

from research.discovery_config import load_signal_discovery_config
from research.discovery_experiment import run_signal_discovery_experiment
from research.storage.discovery_repository import DiscoveryRepository

ROOT = Path(__file__).resolve().parents[2]


JUMP_TS = 60_000  # long flat runway so the control-matching +/-10s exclusion window is a small
                  # slice of the market, not most of it (real markets last 900s; this fixture
                  # needs to be long enough for the exclusion window to actually leave a pool)


def _build_fixture_db(tmp_path):
    rec = make_recorder(tmp_path)
    insert_rows(rec, "markets", [market_row("M_TRIG", end_ts=JUMP_TS + 10_000)])
    # Long flat runway, then a sharp upward jump -- a real momentum+flow+
    # imbalance trigger, same shape as the TP/SL pipeline's own e2e fixture
    # (tests/e2e/test_research_pipeline_e2e.py), just with more history
    # before the jump so the resampled-bucket sigma has enough samples.
    insert_rows(rec, "binance_book", [
        binance_book_row("M_TRIG", ts, [(100.0, 50.0)], [(100.02, 50.0)]) for ts in range(0, JUMP_TS + 100, 100)
    ] + [binance_book_row("M_TRIG", JUMP_TS + 100, [(104.9, 400.0), (104.8, 400.0)], [(105.0, 20.0)])])
    insert_rows(rec, "binance_trades", [
        binance_trade_row("M_TRIG", 1, JUMP_TS, 100.0, 1.0, "SELL"),
        binance_trade_row("M_TRIG", 2, JUMP_TS + 50, 104.9, 5.0, "BUY"),
        binance_trade_row("M_TRIG", 3, JUMP_TS + 90, 105.0, 5.0, "BUY"),
    ])
    insert_rows(rec, "polymarket_book", [
        poly_book_row("M_TRIG", 400, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)]),
        poly_book_row("M_TRIG", JUMP_TS + 100, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)]),
        poly_book_row("M_TRIG", JUMP_TS + 340, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)], up_asks=[(0.50, 100)]),
        poly_book_row("M_TRIG", JUMP_TS + 360, up_bid=0.49, up_ask=0.50, up_bids=[(0.49, 100)],
                       up_asks=[(0.50, 20), (0.51, 30), (0.52, 100)]),
        poly_book_row("M_TRIG", JUMP_TS + 900, up_bid=0.56, up_ask=0.58, up_bids=[(0.56, 100)], up_asks=[(0.58, 100)]),
        poly_book_row("M_TRIG", JUMP_TS + 2_100, up_bid=0.57, up_ask=0.59, up_bids=[(0.57, 100)], up_asks=[(0.59, 100)]),
    ])
    rec.close()
    return tmp_path / "test.db"


def _write_config(tmp_path, db_path, results_db) -> Path:
    config_path = tmp_path / "signal_discovery.yaml"
    config_path.write_text(
        f"""
experiment:
  name: e2e_signal_discovery
data:
  BTC: {db_path}
results_db: {results_db}
latency_grid_ms: [250]
size_grid_shares: [100]
controls_per_signal: 3
bootstrap:
  iterations: 20
  seed: 1337
ranking_horizon_ms: 1000
signal_grid:
  momentum_window_ms: [100, 200]
  volatility_window_ms: [1000]
  z_threshold: [2.0]
  flow_window_ms: [500]
  flow_threshold: [0.3]
  max_spread: [0.05]
  min_contract_price: [0.05]
  max_contract_price: [0.95]
""",
        encoding="utf-8",
    )
    return config_path


def test_signal_discovery_e2e_in_process(tmp_path):
    db_path = _build_fixture_db(tmp_path)
    results_db = tmp_path / "sr_results.db"
    config_path = _write_config(tmp_path, db_path, results_db)

    config = load_signal_discovery_config(str(config_path))
    result = run_signal_discovery_experiment(config)

    assert result.signal_configs == 2  # 2 momentum values, everything else singleton
    assert result.signals > 0
    assert result.executable_entries > 0
    # controls.py's own matching logic (determinism, exclusion window,
    # regime filtering, ranking) is unit-tested rigorously in
    # tests/test_signal_discovery_controls.py. This tiny fixture has only
    # ONE volatility event (the trigger jump itself) in its whole timeline,
    # so there is no genuinely regime-matching "quiet" moment elsewhere for
    # it to be matched against -- zero controls here is the CORRECT,
    # honest answer (never fabricate a mismatched control), not a bug.

    with DiscoveryRepository(str(results_db)) as repo:
        conn = repo.conn
        status = conn.execute(
            "SELECT status FROM experiments WHERE experiment_id=?", (result.experiment_id,)
        ).fetchone()[0]
        assert status == "SUCCESS"

        assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == result.signals
        assert conn.execute("SELECT COUNT(*) FROM entry_marks").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM signal_response").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM signal_path_stats").fetchone()[0] > 0
        # controls/control_response counts are legitimately 0 on this
        # single-volatility-event fixture -- see the comment above.
        assert conn.execute("SELECT COUNT(*) FROM signal_config_summary").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM signal_response_summary").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM signal_hit_summary").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM signal_first_passage_summary").fetchone()[0] > 0

        # checkpoints: both configs marked COMPLETE
        checkpoints = conn.execute(
            "SELECT status FROM checkpoints WHERE experiment_id=?", (result.experiment_id,)
        ).fetchall()
        assert len(checkpoints) == 2
        assert all(c["status"] == "COMPLETE" for c in checkpoints)

        # plateau pass ran without error (it needs zero controls -> zero
        # uplift -> plateau_score legitimately NULL here; the scoring
        # formula itself is unit-tested in test_signal_discovery_robustness.py)
        conn.execute(
            "SELECT plateau_score FROM signal_config_summary WHERE experiment_id=? AND asset IS NULL",
            (result.experiment_id,),
        ).fetchall()

        # one real signal's full chain should be hand-traceable (contract #50):
        signal_row = conn.execute("SELECT * FROM signals LIMIT 1").fetchone()
        entry_row = conn.execute(
            "SELECT * FROM entry_marks WHERE signal_id=?", (signal_row["signal_id"],)
        ).fetchone()
        assert entry_row is not None
        if entry_row["status"] == "EXECUTED":
            response_row = conn.execute(
                "SELECT * FROM signal_response WHERE entry_mark_id=?", (entry_row["entry_mark_id"],)
            ).fetchone()
            assert response_row is not None
            if response_row["status"] == "AVAILABLE":
                assert abs(
                    response_row["raw_response"] - (response_row["future_sell_vwap"] - entry_row["entry_vwap"])
                ) < 1e-9


def test_signal_discovery_resume_skips_complete_configs(tmp_path):
    db_path = _build_fixture_db(tmp_path)
    results_db = tmp_path / "sr_results.db"
    config_path = _write_config(tmp_path, db_path, results_db)
    config = load_signal_discovery_config(str(config_path))

    result_1 = run_signal_discovery_experiment(config)
    result_2 = run_signal_discovery_experiment(config)  # same deterministic experiment_id -> resume

    assert result_1.experiment_id == result_2.experiment_id
    with DiscoveryRepository(str(results_db)) as repo:
        conn = repo.conn
        signal_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        # resume must not duplicate rows (discard-then-recompute is idempotent)
        assert signal_count == result_1.signals


def _run_cli(args, cwd):
    env = os.environ.copy()
    return subprocess.run([sys.executable, "-m", "research.cli", *args], cwd=cwd, env=env,
                           text=True, capture_output=True, check=False)


def test_signal_discovery_cli_commands(tmp_path):
    db_path = _build_fixture_db(tmp_path)
    results_db = tmp_path / "sr_results.db"
    config_path = _write_config(tmp_path, db_path, results_db)

    proc = _run_cli(["run-signal-discovery", str(config_path)], cwd=ROOT)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    experiment_id = None
    for line in proc.stdout.splitlines():
        if line.startswith("experiment_id:"):
            experiment_id = line.split(":", 1)[1].strip()
    assert experiment_id

    top = _run_cli(["top-signals", "--experiment", experiment_id, "--results-db", str(results_db),
                     "--horizon-ms", "1000"], cwd=ROOT)
    assert top.returncode == 0, top.stderr

    with DiscoveryRepository(str(results_db)) as repo:
        signal_id = repo.conn.execute("SELECT signal_id FROM signals LIMIT 1").fetchone()["signal_id"]
        signal_config_id = repo.conn.execute("SELECT signal_config_id FROM signal_configs LIMIT 1").fetchone()[0]

    inspect_signal = _run_cli(["inspect-signal", signal_id, "--results-db", str(results_db),
                                "--latency", "250", "--size", "100"], cwd=ROOT)
    assert inspect_signal.returncode == 0, inspect_signal.stderr
    assert "SIGNAL" in inspect_signal.stdout
    assert "ENTRY" in inspect_signal.stdout
    assert "RESPONSES" in inspect_signal.stdout

    inspect_cfg = _run_cli(["inspect-signal-config", signal_config_id, "--experiment", experiment_id,
                             "--results-db", str(results_db)], cwd=ROOT)
    assert inspect_cfg.returncode == 0, inspect_cfg.stderr
    assert "CONFIG" in inspect_cfg.stdout
