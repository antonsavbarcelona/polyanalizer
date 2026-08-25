from research.discovery_grid import generate_signal_discovery_configs


def test_grid_generates_paired_axes_only():
    grid = {
        "momentum_window_ms": [500, 1_000],
        "volatility_window_ms": [30_000],
        "z_threshold": [2.0, 2.5],
        "flow_window_ms": [None, 1_000],
        "flow_threshold": [None, 0.3],
    }
    configs = generate_signal_discovery_configs(grid)
    # 2 momentum x 1 vol x 2 z x (flow: only (None,None) and (1000,0.3) survive pairing) = 2*1*2*2 = 8
    assert len(configs) == 8
    for cfg in configs:
        assert (cfg.flow_window_ms is None) == (cfg.flow_threshold is None)
        assert cfg.momentum_type == "RETURN"

    # deterministic ids
    configs2 = generate_signal_discovery_configs(grid)
    assert [c.id for c in configs] == [c.id for c in configs2]
    assert len({c.id for c in configs}) == 8  # all distinct
