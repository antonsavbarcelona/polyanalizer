"""YAML config for the signal-response discovery pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research.config import ConfigError, load_config_dict  # reuse the existing tiny YAML loader
from research.discovery.response import HORIZONS_MS


@dataclass(frozen=True)
class SignalDiscoveryExperimentConfig:
    name: str
    data: dict[str, str]
    results_db: str = "data/signal_discovery_results.db"
    signal_grid: dict[str, list[Any]] = field(default_factory=dict)
    latency_grid_ms: tuple[int, ...] = (100, 250, 500)
    size_grid_shares: tuple[float, ...] = (25.0, 100.0, 500.0)
    controls_per_signal: int = 5
    bootstrap_iterations: int = 1_000
    bootstrap_seed: int = 1337
    ranking_horizon_ms: int = 2_000
    max_markets_per_asset: int | None = None

    def __post_init__(self) -> None:
        if self.ranking_horizon_ms not in HORIZONS_MS:
            raise ConfigError(
                f"ranking_horizon_ms={self.ranking_horizon_ms} is not one of the fixed horizons {HORIZONS_MS}"
            )


def load_signal_discovery_config(path: str | Path) -> SignalDiscoveryExperimentConfig:
    raw = load_config_dict(path)
    try:
        experiment = raw["experiment"]
        data = {str(asset).upper(): str(db_path) for asset, db_path in raw["data"].items()}
        signal_grid = raw.get("signal_grid")
        if not isinstance(signal_grid, dict):
            raise ConfigError("signal_grid must be a mapping")
        return SignalDiscoveryExperimentConfig(
            name=str(experiment["name"]),
            data=data,
            results_db=str(raw.get("results_db", "data/signal_discovery_results.db")),
            signal_grid=signal_grid,
            latency_grid_ms=tuple(int(v) for v in raw.get("latency_grid_ms", [100, 250, 500])),
            size_grid_shares=tuple(float(v) for v in raw.get("size_grid_shares", [25.0, 100.0, 500.0])),
            controls_per_signal=int(raw.get("controls_per_signal", 5)),
            bootstrap_iterations=int(raw.get("bootstrap", {}).get("iterations", 1_000)),
            bootstrap_seed=int(raw.get("bootstrap", {}).get("seed", 1337)),
            ranking_horizon_ms=int(raw.get("ranking_horizon_ms", 2_000)),
            max_markets_per_asset=(
                int(raw["max_markets_per_asset"]) if raw.get("max_markets_per_asset") is not None else None
            ),
        )
    except KeyError as exc:
        raise ConfigError(f"missing required config key: {exc.args[0]}") from exc
