"""Deterministic grid generation for SignalDiscoveryConfig. Mirrors
research/grid.py's swept-vs-singleton convention: momentum_window_ms and
the four paired axes (volatility/z, flow, volume, imbalance) are full
Cartesian-product swept; everything else is a singleton (first value only)
-- sweep those separately via multiple config files/runs if needed."""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

from research.config import ConfigError
from research.grid import _first, _optional_float, _optional_int
from research.discovery_types import SignalDiscoveryConfig, deterministic_id


def generate_signal_discovery_configs(grid: dict[str, list[Any]]) -> list[SignalDiscoveryConfig]:
    if "momentum_window_ms" not in grid:
        raise ConfigError("signal_grid.momentum_window_ms is required")

    vol_windows = grid.get("volatility_window_ms", [None])
    z_thresholds = grid.get("z_threshold", [None])
    abs_move_bps = grid.get("absolute_move_threshold_bps", [None])
    flow_windows = grid.get("flow_window_ms", [None])
    flow_thresholds = grid.get("flow_threshold", [None])
    volume_windows = grid.get("volume_window_ms", [None])
    volume_z_thresholds = grid.get("volume_z_threshold", [None])
    imbalance_depths = grid.get("imbalance_depth", [None])
    imbalance_thresholds = grid.get("imbalance_threshold", [None])

    out: list[SignalDiscoveryConfig] = []
    for values in product(
        grid["momentum_window_ms"], vol_windows, z_thresholds, abs_move_bps,
        flow_windows, flow_thresholds, volume_windows, volume_z_thresholds,
        imbalance_depths, imbalance_thresholds,
    ):
        (momentum, vol_window, z, bps, flow_window, flow_thr, volume_window, volume_z_thr,
         imb_depth, imb_thr) = values

        if (vol_window is None) != (z is None):
            continue
        if (flow_window is None) != (flow_thr is None):
            continue
        if (volume_window is None) != (volume_z_thr is None):
            continue
        if (imb_depth is None) != (imb_thr is None):
            continue
        if z is None and bps is None:
            continue  # SignalDiscoveryConfig requires at least one trigger

        cfg = SignalDiscoveryConfig(
            id="",
            momentum_window_ms=int(momentum),
            volatility_window_ms=int(vol_window) if vol_window is not None else None,
            z_threshold=float(z) if z is not None else None,
            absolute_move_threshold_bps=float(bps) if bps is not None else None,
            flow_window_ms=int(flow_window) if flow_window is not None else None,
            flow_threshold=float(flow_thr) if flow_thr is not None else None,
            volume_window_ms=int(volume_window) if volume_window is not None else None,
            volume_z_threshold=float(volume_z_thr) if volume_z_thr is not None else None,
            imbalance_depth=int(imb_depth) if imb_depth is not None else None,
            imbalance_threshold=float(imb_thr) if imb_thr is not None else None,
            poly_lag_window_ms=_optional_int(grid, "poly_lag_window_ms"),
            max_poly_reprice=_optional_float(grid, "max_poly_reprice"),
            min_contract_price=_optional_float(grid, "min_contract_price"),
            max_contract_price=_optional_float(grid, "max_contract_price"),
            max_spread=_optional_float(grid, "max_spread"),
            min_time_remaining_ms=_optional_int(grid, "min_time_remaining_ms"),
            max_time_remaining_ms=_optional_int(grid, "max_time_remaining_ms"),
            reset_threshold=float(_first(grid.get("reset_threshold", [1.0]))),
            cooldown_ms=int(_first(grid.get("cooldown_ms", [3_000]))),
        )
        out.append(replace(cfg, id=deterministic_id("sr_sigcfg", cfg)))

    if not out:
        raise ConfigError("signal_grid produced no valid SignalDiscoveryConfig")
    return out
