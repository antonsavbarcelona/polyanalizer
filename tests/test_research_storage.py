import json

from research.analytics.bootstrap import bootstrap_confidence_intervals
from research.analytics.metrics import compute_strategy_metrics, metrics_by_asset, metrics_by_market
from research.storage.results_repository import ResultsRepository
from research.types import ExecutionConfig, ExitConfig, SignalConfig, StrategyConfig, TradeResult


def _trade(strategy_id, signal_id="s1", market_id="m1", asset="BTC", net=0.02):
    return TradeResult(
        strategy_id=strategy_id,
        signal_id=signal_id,
        asset=asset,
        market_id=market_id,
        direction="UP",
        signal_ts=1_000,
        entry_requested_ts=1_100,
        entry_actual_ts=1_120,
        entry_vwap=0.53,
        exit_ts=2_000,
        exit_price=0.57,
        exit_reason="TP",
        holding_ms=880,
        gross_pnl_per_share=0.04,
        fees_per_share=0.04 - net,
        net_pnl_per_share=net,
        pnl_total=net * 100,
        mfe=0.04,
        mae=-0.01,
        tp_hit=True,
        sl_hit=False,
        ambiguous_exit=False,
    )


def _strategy():
    signal = SignalConfig(
        id="sig_h1",
        momentum_window_ms=1_000,
        volatility_window_ms=120_000,
        z_threshold=2.5,
        flow_window_ms=1_000,
        flow_threshold=0.25,
    )
    execution = ExecutionConfig(id="lat250_size100", latency_ms=250, size_shares=100)
    exit_cfg = ExitConfig.absolute_delta("tp4_sl3_hold10s", tp_delta=0.04, sl_delta=0.03, max_holding_ms=10_000)
    return StrategyConfig(signal=signal, execution=execution, exit=exit_cfg)


def test_results_repository_saves_experiment_configs_and_strategy():
    strategy = _strategy()
    with ResultsRepository(":memory:") as repo:
        experiment_id = repo.create_experiment(
            "unit",
            {"assets": ["BTC"]},
            data_sources=["data/analyzer_btc.db"],
            data_fingerprint="fp",
            code_version="test",
            started_at=123,
        )
        strategy_id = repo.save_strategy_config(strategy)

        conn = repo.conn
        assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
        assert conn.execute("SELECT name FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()[0] == "unit"
        assert conn.execute("SELECT COUNT(*) FROM signal_configs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM execution_configs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM exit_configs").fetchone()[0] == 1
        assert conn.execute("SELECT strategy_id FROM strategies").fetchone()[0] == strategy_id
        assert strategy_id == strategy.strategy_id()


def test_results_repository_saves_trades_metrics_and_bootstrap_rows():
    strategy_id = _strategy().strategy_id()
    rows = [
        _trade(strategy_id, "s1", "m1", "BTC", 0.02),
        _trade(strategy_id, "s2", "m2", "ETH", -0.01),
    ]

    with ResultsRepository(":memory:") as repo:
        for row in rows:
            repo.save_trade_result(row)
        repo.save_strategy_metrics(compute_strategy_metrics(strategy_id, rows))
        for metrics in metrics_by_asset(strategy_id, rows):
            repo.save_strategy_metrics(metrics)
        for metrics in metrics_by_market(strategy_id, rows):
            repo.save_strategy_metrics(metrics)
        for result in bootstrap_confidence_intervals(rows, strategy_id, iterations=10, seed=2):
            repo.save_bootstrap_result(result)

        conn = repo.conn
        assert conn.execute("SELECT COUNT(*) FROM trade_results").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM strategy_metrics").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM strategy_metrics_by_asset").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM strategy_metrics_by_market").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM bootstrap_results").fetchone()[0] == 2

        payload = conn.execute("SELECT payload_json FROM strategy_metrics").fetchone()[0]
        decoded = json.loads(payload)
        assert decoded["strategy_id"] == strategy_id
        assert decoded["trade_count"] == 2
