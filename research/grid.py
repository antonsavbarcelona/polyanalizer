"""Deterministic grid generation for research experiments."""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

from research.config import ConfigError, ExperimentConfig
from research.types import ExecutionConfig, ExitConfig, SignalConfig, StrategyConfig, deterministic_id


def generate_signal_configs(config: ExperimentConfig) -> list[SignalConfig]:
    grid = config.signal_grid
    required = ["momentum_window_ms", "volatility_window_ms", "z_threshold"]
    for key in required:
        if key not in grid:
            raise ConfigError(f"signal_grid.{key} is required")

    flow_windows = grid.get("flow_window_ms", [None])
    flow_thresholds = grid.get("flow_threshold", [None])
    out: list[SignalConfig] = []
    for values in product(
        grid["momentum_window_ms"],
        grid["volatility_window_ms"],
        grid["z_threshold"],
        flow_windows,
        flow_thresholds,
        grid.get("imbalance_depth", [None]),
        grid.get("imbalance_threshold", [None]),
    ):
        momentum, vol, z, flow_window, flow_threshold, depth, imbalance_threshold = values
        if (flow_window is None) != (flow_threshold is None):
            continue
        if (depth is None) != (imbalance_threshold is None):
            continue
        cfg = SignalConfig(
            id="",
            momentum_window_ms=int(momentum),
            volatility_window_ms=int(vol),
            z_threshold=float(z),
            flow_window_ms=int(flow_window) if flow_window is not None else None,
            flow_threshold=float(flow_threshold) if flow_threshold is not None else None,
            imbalance_depth=int(depth) if depth is not None else None,
            imbalance_threshold=float(imbalance_threshold) if imbalance_threshold is not None else None,
            poly_lag_window_ms=_optional_int(grid, "poly_lag_window_ms"),
            max_poly_reprice=_optional_float(grid, "max_poly_reprice"),
            max_spread=_optional_float(grid, "max_spread"),
            min_contract_price=_optional_float(grid, "min_contract_price"),
            max_contract_price=_optional_float(grid, "max_contract_price"),
            min_time_remaining_ms=_optional_int(grid, "min_time_remaining_ms"),
            max_time_remaining_ms=_optional_int(grid, "max_time_remaining_ms"),
            reset_z=float(_first(grid.get("reset_z", [1.0]))),
            cooldown_ms=int(_first(grid.get("cooldown_ms", [3_000]))),
        )
        out.append(replace(cfg, id=deterministic_id("sigcfg", cfg)))
    if not out:
        raise ConfigError("signal_grid produced no valid SignalConfig")
    return out


def generate_execution_configs(config: ExperimentConfig) -> list[ExecutionConfig]:
    grid = config.execution_grid
    for key in ["latency_ms", "size_shares"]:
        if key not in grid:
            raise ConfigError(f"execution_grid.{key} is required")
    out = []
    for latency, size in product(grid["latency_ms"], grid["size_shares"]):
        cfg = ExecutionConfig(id="", latency_ms=int(latency), size_shares=float(size))
        out.append(replace(cfg, id=deterministic_id("execcfg", cfg)))
    if not out:
        raise ConfigError("execution_grid produced no ExecutionConfig")
    return out


def generate_exit_configs(config: ExperimentConfig) -> list[ExitConfig]:
    grid = config.exit_grid
    out = []
    for tp, sl, hold in product(
        grid.get("tp_delta", [None]),
        grid.get("sl_delta", [None]),
        grid.get("max_holding_ms", [None]),
    ):
        if tp is None and sl is None and hold is None:
            continue
        cfg = ExitConfig.absolute_delta(
            id="",
            tp_delta=float(tp) if tp is not None else None,
            sl_delta=float(sl) if sl is not None else None,
            max_holding_ms=int(hold) if hold is not None else None,
        )
        out.append(replace(cfg, id=deterministic_id("exitcfg", cfg)))
    if not out:
        raise ConfigError("exit_grid produced no ExitConfig")
    return out


def generate_strategy_configs(config: ExperimentConfig) -> list[StrategyConfig]:
    return [
        StrategyConfig(signal=signal, execution=execution, exit=exit_cfg)
        for signal, execution, exit_cfg in product(
            generate_signal_configs(config),
            generate_execution_configs(config),
            generate_exit_configs(config),
        )
    ]


def _first(values: list[Any]) -> Any:
    return values[0] if isinstance(values, list) else values


def _optional_int(grid: dict[str, list[Any]], key: str) -> int | None:
    if key not in grid:
        return None
    value = _first(grid[key])
    return int(value) if value is not None else None


def _optional_float(grid: dict[str, list[Any]], key: str) -> float | None:
    if key not in grid:
        return None
    value = _first(grid[key])
    return float(value) if value is not None else None
