import math

import pytest

from research.analytics.bootstrap import bootstrap_confidence_intervals, bootstrap_metric_samples
from research.analytics.metrics import compute_strategy_metrics, metrics_by_asset, metrics_by_market
from research.types import TradeResult


def _trade(
    signal_id,
    market_id,
    net,
    *,
    gross=None,
    asset="BTC",
    exit_reason="TP",
    tp_hit=False,
    sl_hit=False,
):
    gross = net if gross is None else gross
    return TradeResult(
        strategy_id="S1",
        signal_id=signal_id,
        asset=asset,
        market_id=market_id,
        direction="UP",
        signal_ts=0,
        entry_requested_ts=100,
        entry_actual_ts=100,
        entry_vwap=0.50,
        exit_ts=200,
        exit_price=0.50 + gross,
        exit_reason=exit_reason,
        holding_ms=100,
        gross_pnl_per_share=gross,
        fees_per_share=gross - net,
        net_pnl_per_share=net,
        pnl_total=net * 100,
        mfe=max(net, 0.0),
        mae=min(net, 0.0),
        tp_hit=tp_hit,
        sl_hit=sl_hit,
        ambiguous_exit=False,
    )


def test_metrics_use_net_pnl_for_winrate_not_tp_hit():
    rows = [
        _trade("s1", "m1", 0.02, tp_hit=True),
        _trade("s2", "m1", -0.01, exit_reason="TP", tp_hit=True),
        _trade("s3", "m2", 0.03, exit_reason="TIMEOUT"),
    ]

    metrics = compute_strategy_metrics("S1", rows, signal_count=4, entry_count=3)

    assert metrics.trade_count == 3
    assert metrics.not_executable_count == 1
    assert metrics.win_count == 2
    assert metrics.loss_count == 1
    assert metrics.win_rate == pytest.approx(2 / 3)
    assert metrics.tp_hit_rate == pytest.approx(2 / 3)
    assert metrics.timeout_rate == pytest.approx(1 / 3)
    assert metrics.mean_net_pnl == pytest.approx((0.02 - 0.01 + 0.03) / 3)
    assert metrics.median_net_pnl == pytest.approx(0.02)
    assert metrics.avg_win == pytest.approx(0.025)
    assert metrics.avg_loss == pytest.approx(-0.01)
    assert metrics.payoff_ratio == pytest.approx(2.5)
    assert metrics.profit_factor == pytest.approx(5.0)


def test_metrics_profit_factor_is_infinite_when_no_losses():
    metrics = compute_strategy_metrics("S1", [_trade("s1", "m1", 0.01)])

    assert math.isinf(metrics.profit_factor)


def test_metrics_group_by_asset_and_market():
    rows = [
        _trade("s1", "m1", 0.01, asset="BTC"),
        _trade("s2", "m2", -0.01, asset="ETH"),
        _trade("s3", "m2", 0.03, asset="ETH"),
    ]

    by_asset = metrics_by_asset("S1", rows)
    by_market = metrics_by_market("S1", rows)

    assert [(m.asset, m.trade_count) for m in by_asset] == [("BTC", 1), ("ETH", 2)]
    assert [(m.market_id, m.trade_count) for m in by_market] == [("m1", 1), ("m2", 2)]


def test_bootstrap_samples_whole_markets_not_individual_trades():
    rows = [
        _trade("a1", "A", 10.0),
        _trade("a2", "A", 10.0),
        _trade("b1", "B", -10.0),
        _trade("b2", "B", -10.0),
    ]

    samples = bootstrap_metric_samples(rows, "mean_net_pnl", iterations=100, seed=7)

    assert set(samples).issubset({-10.0, 0.0, 10.0})
    assert samples


def test_bootstrap_confidence_intervals_report_mean_and_winrate():
    rows = [
        _trade("a1", "A", 0.02),
        _trade("a2", "A", 0.01),
        _trade("b1", "B", -0.01),
    ]

    results = bootstrap_confidence_intervals(rows, "S1", iterations=50, seed=1)

    assert [result.metric for result in results] == ["mean_net_pnl", "win_rate"]
    assert all(result.sampling_unit == "market_id" for result in results)
    assert all(result.low is not None and result.high is not None for result in results)
    assert all(result.low <= result.high for result in results)
