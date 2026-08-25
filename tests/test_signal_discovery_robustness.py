"""Plateau/neighbor, ablation, and overlap (IMPLEMENTATION CONTRACT #35-38)."""
from __future__ import annotations

from research.discovery.ablation import FEATURE_STEPS, generate_ablation_sequence
from research.discovery.overlap import compute_overlap
from research.discovery.plateau import compute_plateau_metrics, find_neighbors, is_neighbor
from research.discovery_types import SignalDiscoveryConfig, SignalSnapshot


def _cfg(id_, momentum, z, flow_window, flow_thr) -> SignalDiscoveryConfig:
    return SignalDiscoveryConfig(
        id=id_, momentum_window_ms=momentum, volatility_window_ms=30_000, z_threshold=z,
        flow_window_ms=flow_window, flow_threshold=flow_thr,
    )


GRID = {"momentum_window_ms": [500, 1_000], "z_threshold": [2.0, 2.5, 3.0], "flow_threshold": [0.2, 0.3, 0.4]}


def test_neighbor_requires_exactly_one_axis_one_step():
    center = _cfg("center", 500, 2.5, 1_000, 0.3)

    # differs only in z_threshold, by one adjacent grid step (2.5 -> 2.75... but grid has 2.0/2.5/3.0, so 3.0 is the adjacent step)
    z_neighbor = _cfg("z_neighbor", 500, 3.0, 1_000, 0.3)
    assert is_neighbor(center, z_neighbor, GRID)

    # differs only in flow_threshold, by one adjacent step (0.3 -> 0.4)
    flow_neighbor = _cfg("flow_neighbor", 500, 2.5, 1_000, 0.4)
    assert is_neighbor(center, flow_neighbor, GRID)

    # differs in z AND flow at once -> not a neighbor even though both are "close"
    two_axis = _cfg("two_axis", 500, 3.0, 1_000, 0.4)
    assert not is_neighbor(center, two_axis, GRID)

    # differs in z by TWO grid steps (2.5 -> not adjacent to nothing else present skip) -- use a config two steps away conceptually
    far = _cfg("far", 500, 2.0, 1_000, 0.2)  # differs on BOTH z and flow -> not neighbor regardless of step size
    assert not is_neighbor(center, far, GRID)

    assert find_neighbors(center, [z_neighbor, flow_neighbor, two_axis, far], GRID) == [z_neighbor, flow_neighbor]


def test_plateau_score_rewards_smooth_uplift_and_penalizes_scatter():
    smooth = compute_plateau_metrics(central_uplift_mean=1.5, signal_count=100,
                                      neighbor_uplifts=[1.2, 1.4, 1.6, 1.3])
    spike = compute_plateau_metrics(central_uplift_mean=3.8, signal_count=40,
                                     neighbor_uplifts=[-0.1, 0.0, 0.1, -0.2])

    assert smooth.neighbor_positive_ratio == 1.0
    assert spike.neighbor_positive_ratio == 0.25
    assert smooth.plateau_score > 0
    assert spike.plateau_score is not None
    # A lonely, unsupported spike must not automatically outrank a smooth,
    # neighbor-confirmed region -- that's the entire point of the score.
    assert smooth.plateau_score > spike.plateau_score


def test_plateau_score_none_without_central_uplift():
    m = compute_plateau_metrics(central_uplift_mean=None, signal_count=50, neighbor_uplifts=[1.0, 2.0])
    assert m.plateau_score is None


def test_ablation_sequence_never_uses_sentinel_disabled():
    base = SignalDiscoveryConfig(
        id="base", momentum_window_ms=500, volatility_window_ms=30_000, z_threshold=2.5,
        flow_window_ms=1_000, flow_threshold=0.3, imbalance_depth=5, imbalance_threshold=0.1,
        volume_window_ms=2_000, volume_z_threshold=1.5, poly_lag_window_ms=500, max_poly_reprice=0.01,
    )
    seq = generate_ablation_sequence(base)
    assert len(seq) == len(FEATURE_STEPS)

    momentum_only = seq[0]
    assert momentum_only.flow_threshold is None and momentum_only.flow_window_ms is None
    assert momentum_only.imbalance_threshold is None
    assert momentum_only.volume_z_threshold is None
    assert momentum_only.max_poly_reprice is None
    assert momentum_only.momentum_window_ms == 500  # the always-on baseline

    plus_flow = seq[1]
    assert plus_flow.flow_threshold == 0.3 and plus_flow.flow_window_ms == 1_000
    assert plus_flow.imbalance_threshold is None  # not yet enabled

    full = seq[-1]
    assert full.flow_threshold == 0.3
    assert full.imbalance_threshold == 0.1
    assert full.volume_z_threshold == 1.5
    assert full.max_poly_reprice == 0.01

    # every step must be internally valid (no sentinel, paired fields consistent)
    for cfg in seq:
        assert (cfg.flow_window_ms is None) == (cfg.flow_threshold is None)
        assert (cfg.imbalance_depth is None) == (cfg.imbalance_threshold is None)
        assert (cfg.volume_window_ms is None) == (cfg.volume_z_threshold is None)

    # deterministic ids: rerunning produces byte-identical config ids
    seq2 = generate_ablation_sequence(base)
    assert [c.id for c in seq] == [c.id for c in seq2]


def _sig(mid, direction, ts) -> SignalSnapshot:
    return SignalSnapshot(
        signal_id=f"{mid}_{ts}", signal_config_id="X", asset="BTC", market_id=mid, direction=direction,
        signal_ts=ts, binance_mid=None, return_100ms=None, return_250ms=None, return_500ms=None,
        return_750ms=None, return_1s=None, return_1_5s=None, return_2s=None, return_3s=None, return_5s=None,
        return_10s=None, active_momentum_value=None, vol_10s=None, vol_30s=None, vol_60s=None, vol_120s=None,
        vol_300s=None, active_volatility=None, z_score=None, flow_100ms=None, flow_250ms=None, flow_500ms=None,
        flow_750ms=None, flow_1s=None, flow_2s=None, flow_3s=None, flow_5s=None, flow_10s=None, active_flow=None,
        volume_z=None, imbalance_top1=None, imbalance_top3=None, imbalance_top5=None, imbalance_top10=None,
        microprice_bias=None, target_bid=None, target_ask=None, target_spread=None, target_bid_size=None,
        target_ask_size=None, poly_move_100ms=None, poly_move_250ms=None, poly_move_500ms=None,
        poly_move_1s=None, poly_move_2s=None, time_remaining_ms=None, time_elapsed_ms=None,
        reference_price=None, chainlink_twap=None, distance_to_reference_bps=None, volatility_regime=None,
        tte_regime="600s+", price_regime=None, spread_regime=None,
    )


def test_overlap_symmetric_matching():
    a = [_sig("M1", "UP", 1_000), _sig("M1", "UP", 5_000), _sig("M2", "DOWN", 9_000)]
    b = [_sig("M1", "UP", 1_100), _sig("M2", "UP", 5_000), _sig("M3", "UP", 1_000)]
    # a[0](M1,UP,1000) matches b[0](M1,UP,1100): |diff|=100<=250 -> match.
    # a[1](M1,UP,5000): no candidate in b at M1/UP near 5000 -> no match.
    # a[2](M2,DOWN,9000): b has no M2/DOWN at all -> no match.
    overlap = compute_overlap(a, b)
    assert abs(overlap - 1 / 3) < 1e-9  # 1 matched / min(3,3)


def test_overlap_empty_inputs():
    assert compute_overlap([], [_sig("M1", "UP", 0)]) == 0.0
    assert compute_overlap([_sig("M1", "UP", 0)], []) == 0.0
