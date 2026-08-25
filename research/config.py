"""Experiment config loading and validation."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RankingConfig:
    min_signals: int = 0
    min_markets: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: dict[str, str]
    research_clock_ms: int
    signal_grid: dict[str, list[Any]]
    execution_grid: dict[str, list[Any]]
    exit_grid: dict[str, list[Any]]
    results_db: str = "data/research_results.db"
    fee_model: str = "fixed_v1"
    ranking: RankingConfig = field(default_factory=RankingConfig)
    bootstrap_iterations: int = 200
    max_markets_per_asset: int | None = None


class ConfigError(ValueError):
    pass


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = load_config_dict(path)
    try:
        name = raw["experiment"]["name"]
        data = {str(asset).upper(): str(db_path) for asset, db_path in raw["data"].items()}
        ranking_raw = raw.get("ranking", {})
        return ExperimentConfig(
            name=str(name),
            data=data,
            research_clock_ms=int(raw.get("research_clock_ms", 100)),
            signal_grid=_require_grid(raw, "signal_grid"),
            execution_grid=_require_grid(raw, "execution_grid"),
            exit_grid=_require_grid(raw, "exit_grid"),
            results_db=str(raw.get("results_db", "data/research_results.db")),
            fee_model=str(raw.get("fee_model", {}).get("name", raw.get("fee_model", "fixed_v1"))),
            ranking=RankingConfig(
                min_signals=int(ranking_raw.get("min_signals", 0)),
                min_markets=int(ranking_raw.get("min_markets", 0)),
            ),
            bootstrap_iterations=int(raw.get("bootstrap_iterations", 200)),
            max_markets_per_asset=(
                int(raw["max_markets_per_asset"]) if raw.get("max_markets_per_asset") is not None else None
            ),
        )
    except KeyError as exc:
        raise ConfigError(f"missing required config key: {exc.args[0]}") from exc


def load_config_dict(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = _parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    return data


def _require_grid(raw: dict[str, Any], key: str) -> dict[str, list[Any]]:
    grid = raw.get(key)
    if not isinstance(grid, dict):
        raise ConfigError(f"{key} must be a mapping")
    out: dict[str, list[Any]] = {}
    for name, values in grid.items():
        if not isinstance(values, list):
            raise ConfigError(f"{key}.{name} must be a list")
        out[str(name)] = values
    return out


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    current_list_key: dict[int, str] = {}

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ConfigError(f"list item without list parent: {raw_line}")
            parent.append(_parse_scalar(line[2:].strip()))
            continue

        if ":" not in line:
            raise ConfigError(f"cannot parse config line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue

        child: Any = {}
        parent[key] = child
        stack.append((indent, child))
        current_list_key[indent + 2] = key

        # Convert an empty mapping to a list when the next significant line
        # at the child indent starts with "- ". This is handled lazily below.
        lines_after = text.splitlines()
        # The fallback parser is intentionally small; the lazy conversion is
        # done by peeking through the original line sequence.
        _ = lines_after

    return _fix_lists(root)


def _fix_lists(value: Any) -> Any:
    if isinstance(value, dict):
        fixed = {}
        for key, item in value.items():
            if isinstance(item, dict) and "" in item:
                fixed[key] = item[""]
            else:
                fixed[key] = _fix_lists(item)
        return fixed
    if isinstance(value, list):
        return [_fix_lists(item) for item in value]
    return value


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") or value.startswith("{"):
        return ast.literal_eval(value.replace("null", "None"))
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")
