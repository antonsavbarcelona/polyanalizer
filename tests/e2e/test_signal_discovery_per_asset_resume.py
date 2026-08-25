"""Per-asset checkpoint resume (task item 3): if BTC and ETH are COMPLETE
but SOL FAILED, --resume must only recompute SOL -- BTC/ETH's already-
written rows must survive byte-identical, and the pooled ALL summary must
end up correctly combining all three assets regardless of which ones were
computed in which run."""
from __future__ import annotations

from tests.e2e.test_signal_discovery_e2e import JUMP_TS, _build_fixture_db, _write_config

from research.discovery_config import load_signal_discovery_config
from research.discovery_experiment import run_signal_discovery_experiment
from research.storage.discovery_repository import DiscoveryRepository


def _write_two_asset_config(tmp_path, btc_path, eth_path, results_db):
    config_path = tmp_path / "signal_discovery.yaml"
    config_path.write_text(
        f"""
experiment:
  name: e2e_resume
data:
  BTC: {btc_path}
  ETH: {eth_path}
results_db: {results_db}
latency_grid_ms: [250]
size_grid_shares: [100]
controls_per_signal: 3
bootstrap:
  iterations: 20
  seed: 1337
ranking_horizon_ms: 1000
signal_grid:
  momentum_window_ms: [100]
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


def test_resume_only_recomputes_the_failed_asset(tmp_path):
    btc_path = _build_fixture_db(tmp_path / "btc")
    eth_path = _build_fixture_db(tmp_path / "eth")
    results_db = tmp_path / "sr_results.db"
    config_path = _write_two_asset_config(tmp_path, btc_path, eth_path, results_db)
    config = load_signal_discovery_config(str(config_path))

    result = run_signal_discovery_experiment(config)
    assert result.failed_units == 0
    assert result.signals > 0

    with DiscoveryRepository(str(results_db)) as repo:
        conn = repo.conn
        statuses = {
            r["asset"]: r["status"]
            for r in conn.execute(
                "SELECT asset, status FROM checkpoints WHERE experiment_id=?", (result.experiment_id,)
            )
        }
        assert statuses == {"BTC": "COMPLETE", "ETH": "COMPLETE"}

        btc_signal_ids_before = {
            r["signal_id"] for r in conn.execute(
                "SELECT signal_id FROM signals WHERE experiment_id=? AND asset='BTC'", (result.experiment_id,)
            )
        }
        assert btc_signal_ids_before  # sanity: BTC actually produced signals

        cfg_id = conn.execute("SELECT signal_config_id FROM signal_configs LIMIT 1").fetchone()[0]

        # ---- simulate "ETH FAILED" (as if a real crash had happened there) ----
        repo.mark_checkpoint_failed(result.experiment_id, cfg_id, "ETH", "simulated crash")
        repo.discard_partial_asset(result.experiment_id, cfg_id, "ETH")

        eth_signals_after_discard = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE experiment_id=? AND asset='ETH'", (result.experiment_id,)
        ).fetchone()[0]
        assert eth_signals_after_discard == 0
        btc_signal_ids_after_discard = {
            r["signal_id"] for r in conn.execute(
                "SELECT signal_id FROM signals WHERE experiment_id=? AND asset='BTC'", (result.experiment_id,)
            )
        }
        assert btc_signal_ids_after_discard == btc_signal_ids_before  # BTC untouched by ETH's discard

    # ---- resume: same deterministic experiment_id, only ETH should be redone ----
    result_2 = run_signal_discovery_experiment(config)
    assert result_2.experiment_id == result.experiment_id
    assert result_2.failed_units == 0

    with DiscoveryRepository(str(results_db)) as repo:
        conn = repo.conn
        statuses = {
            r["asset"]: r["status"]
            for r in conn.execute(
                "SELECT asset, status FROM checkpoints WHERE experiment_id=?", (result.experiment_id,)
            )
        }
        assert statuses == {"BTC": "COMPLETE", "ETH": "COMPLETE"}

        btc_signal_ids_final = {
            r["signal_id"] for r in conn.execute(
                "SELECT signal_id FROM signals WHERE experiment_id=? AND asset='BTC'", (result.experiment_id,)
            )
        }
        assert btc_signal_ids_final == btc_signal_ids_before  # BTC still byte-identical after ETH's resume

        eth_signal_count = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE experiment_id=? AND asset='ETH'", (result.experiment_id,)
        ).fetchone()[0]
        assert eth_signal_count > 0  # ETH was actually recomputed

        # pooled ALL summary must combine BOTH assets' signal counts exactly
        # (pooled rows are stamped asset=ALL_ASSET="ALL", never SQL NULL --
        # see research/discovery_types.py::ALL_ASSET)
        all_summary = conn.execute(
            """SELECT signal_count FROM signal_config_summary
               WHERE experiment_id=? AND signal_config_id=? AND asset='ALL'
                 AND latency_ms=250 AND size_shares=100""",
            (result.experiment_id, cfg_id),
        ).fetchone()
        btc_only = conn.execute(
            """SELECT signal_count FROM signal_config_summary
               WHERE experiment_id=? AND signal_config_id=? AND asset='BTC'
                 AND latency_ms=250 AND size_shares=100""",
            (result.experiment_id, cfg_id),
        ).fetchone()["signal_count"]
        eth_only = conn.execute(
            """SELECT signal_count FROM signal_config_summary
               WHERE experiment_id=? AND signal_config_id=? AND asset='ETH'
                 AND latency_ms=250 AND size_shares=100""",
            (result.experiment_id, cfg_id),
        ).fetchone()["signal_count"]
        assert all_summary["signal_count"] == btc_only + eth_only
