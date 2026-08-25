"""DiscoveryRepository round-trip + checkpoint/resume (IMPLEMENTATION
CONTRACT #41)."""
from __future__ import annotations

from research.discovery_types import DiscoveryEntryMark, SignalDiscoveryConfig, SignalResponse
from research.storage.discovery_repository import DiscoveryRepository
from tests.test_signal_discovery_golden import _snapshot


def test_repository_round_trip(tmp_path):
    repo = DiscoveryRepository(str(tmp_path / "sr.db"))
    repo.connect()
    try:
        experiment_id = repo.create_experiment(
            "test", {"a": 1}, data_sources={"BTC": "x.db"}, data_fingerprint="fp",
            code_version="v1", fee_model_version="fixed_v1", bootstrap_seed=1337, bootstrap_iterations=100,
        )

        cfg = SignalDiscoveryConfig(id="C1", momentum_window_ms=500, absolute_move_threshold_bps=10.0)
        repo.save_signal_config(cfg)

        signal = _snapshot("SIG_1", "M1", "UP", signal_ts=1_000)
        signal.signal_config_id = "C1"  # _snapshot() defaults to a fixture config id; align it with cfg above
        repo.save_signal(experiment_id, signal)

        mark = DiscoveryEntryMark(
            entry_mark_id="EM1", signal_id="SIG_1", latency_ms=250, size_shares=100,
            entry_target_ts=1_250, entry_actual_ts=1_300, entry_delay_after_target_ms=50,
            entry_best_bid=0.49, entry_best_ask=0.51, entry_vwap=0.516, entry_slippage=0.006,
            available_ask_liquidity=100.0, entry_fee_total=1.7, entry_fee_per_share=0.017, status="EXECUTED",
        )
        repo.save_entry_mark(mark)

        response = SignalResponse(
            response_id="R1", entry_mark_id="EM1", signal_id="SIG_1", signal_config_id="C1",
            asset="BTC", market_id="M1", direction="UP", latency_ms=250, size_shares=100, horizon_ms=1_000,
            response_target_ts=2_300, response_actual_ts=2_300, response_delay_ms=0,
            future_best_bid=0.55, future_best_ask=0.60, future_sell_vwap=0.545, available_bid_liquidity=100.0,
            raw_response=0.029, fee_adjusted_response=0.012, response_positive=True,
            fee_adjusted_positive=True, status="AVAILABLE",
        )
        repo.save_signal_response(response)

        conn = repo.conn
        assert conn.execute("SELECT COUNT(*) FROM signal_configs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM entry_marks").fetchone()[0] == 1
        row = conn.execute("SELECT * FROM signal_response WHERE response_id='R1'").fetchone()
        assert row["response_positive"] == 1  # bool stored as 0/1
        assert abs(row["raw_response"] - 0.029) < 1e-9

        # checkpoint lifecycle (grain: experiment x signal_config x asset)
        assert repo.checkpoint_status(experiment_id, "C1", "BTC") is None
        repo.mark_checkpoint_running(experiment_id, "C1", "BTC")
        assert repo.checkpoint_status(experiment_id, "C1", "BTC") == "RUNNING"
        repo.mark_checkpoint_complete(experiment_id, "C1", "BTC")
        assert repo.checkpoint_status(experiment_id, "C1", "BTC") == "COMPLETE"
        assert repo.asset_checkpoint_statuses(experiment_id, "C1") == {"BTC": "COMPLETE"}

        # a DIFFERENT asset's checkpoint is entirely independent
        repo.mark_checkpoint_failed(experiment_id, "C1", "ETH", "boom")
        assert repo.checkpoint_status(experiment_id, "C1", "ETH") == "FAILED"
        assert repo.checkpoint_status(experiment_id, "C1", "BTC") == "COMPLETE"  # untouched

        # discard_partial_asset must cascade-delete only THAT asset's rows,
        # and clear the (now-stale) pooled ALL summary, but never touch
        # other assets' already-COMPLETE data.
        repo.discard_partial_asset(experiment_id, "C1", "BTC")
        assert conn.execute("SELECT COUNT(*) FROM signals WHERE signal_config_id='C1'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM entry_marks WHERE signal_id='SIG_1'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM signal_response WHERE signal_config_id='C1'").fetchone()[0] == 0
        assert repo.checkpoint_status(experiment_id, "C1", "BTC") is None
        assert repo.checkpoint_status(experiment_id, "C1", "ETH") == "FAILED"  # untouched by BTC's discard

        repo.finish_experiment(experiment_id, "SUCCESS")
        status = conn.execute("SELECT status FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()[0]
        assert status == "SUCCESS"
    finally:
        repo.close()
