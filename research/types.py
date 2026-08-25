"""Canonical dataclasses shared by the offline research pipeline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


def canonical_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: canonical_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): canonical_data(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonical_data(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(canonical_data(value), sort_keys=True, separators=(",", ":"))


def deterministic_id(prefix: str, value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class SignalConfig:
    id: str
    momentum_window_ms: int
    volatility_window_ms: int
    z_threshold: float
    flow_window_ms: int | None = None
    flow_threshold: float | None = None
    imbalance_depth: int | None = None
    imbalance_threshold: float | None = None
    poly_lag_window_ms: int | None = None
    max_poly_reprice: float | None = None
    max_spread: float | None = None
    min_contract_price: float | None = None
    max_contract_price: float | None = None
    min_time_remaining_ms: int | None = None
    max_time_remaining_ms: int | None = None
    reset_z: float = 1.0
    cooldown_ms: int = 3_000


@dataclass(frozen=True)
class ExecutionConfig:
    id: str
    latency_ms: int
    size_shares: float
    entry_mode: str = "TAKER"
    partial_fill_policy: str = "REJECT"
    exit_latency_ms: int = 0


@dataclass(frozen=True)
class ExitConfig:
    id: str
    take_profit_type: str | None = None
    take_profit_value: float | None = None
    stop_loss_type: str | None = None
    stop_loss_value: float | None = None
    max_holding_ms: int | None = None
    tp_execution: str = "MAKER"
    stop_execution: str = "TAKER"
    timeout_execution: str = "TAKER"
    stop_exit_latency_ms: int = 0
    hold_to_resolution: bool = False

    @classmethod
    def absolute_delta(
        cls,
        id: str,
        tp_delta: float | None = None,
        sl_delta: float | None = None,
        max_holding_ms: int | None = None,
        **kwargs: Any,
    ) -> "ExitConfig":
        return cls(
            id=id,
            take_profit_type="ABSOLUTE_DELTA" if tp_delta is not None else None,
            take_profit_value=tp_delta,
            stop_loss_type="ABSOLUTE_DELTA" if sl_delta is not None else None,
            stop_loss_value=sl_delta,
            max_holding_ms=max_holding_ms,
            **kwargs,
        )

    def resolve_take_profit(self, entry_price: float) -> float | None:
        return _resolve_exit_price(entry_price, self.take_profit_type, self.take_profit_value, sign=1.0)

    def resolve_stop_loss(self, entry_price: float) -> float | None:
        return _resolve_exit_price(entry_price, self.stop_loss_type, self.stop_loss_value, sign=-1.0)


def _resolve_exit_price(entry_price: float, kind: str | None, value: float | None, sign: float) -> float | None:
    if kind is None or value is None:
        return None
    if kind == "ABSOLUTE_DELTA":
        return entry_price + sign * value
    if kind == "RELATIVE_PERCENT":
        return entry_price * (1.0 + sign * value)
    raise ValueError(f"unsupported exit type: {kind!r}")


@dataclass(frozen=True)
class StrategyConfig:
    signal: SignalConfig
    execution: ExecutionConfig
    exit: ExitConfig

    def strategy_id(self) -> str:
        return deterministic_id("strategy", self)


@dataclass
class TradeResult:
    strategy_id: str
    signal_id: str
    asset: str | None
    market_id: str
    direction: str
    signal_ts: int
    entry_requested_ts: int
    entry_actual_ts: int
    entry_vwap: float
    exit_ts: int | None
    exit_price: float | None
    exit_reason: str
    holding_ms: int
    gross_pnl_per_share: float
    fees_per_share: float
    net_pnl_per_share: float
    pnl_total: float
    mfe: float | None
    mae: float | None
    tp_hit: bool
    sl_hit: bool
    ambiguous_exit: bool
    trade_result_id: str = ""
    entry_fee_per_share: float | None = None
    exit_fee_per_share: float | None = None
    entry_slippage: float | None = None
    exit_slippage: float | None = None


@dataclass
class StrategyMetrics:
    strategy_id: str
    asset: str | None
    market_id: str | None
    signal_count: int
    entry_count: int
    not_executable_count: int
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float | None
    tp_hit_rate: float | None
    sl_hit_rate: float | None
    timeout_rate: float | None
    mean_gross_pnl: float | None
    median_gross_pnl: float | None
    mean_net_pnl: float | None
    median_net_pnl: float | None
    avg_win: float | None
    avg_loss: float | None
    payoff_ratio: float | None
    profit_factor: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    mfe_mean: float | None
    mae_mean: float | None
    holding_time_mean: float | None
    holding_time_median: float | None
    breakeven_count: int = 0
    mean_entry_slippage: float | None = None
    median_entry_slippage: float | None = None
    mean_exit_slippage: float | None = None
    median_exit_slippage: float | None = None
    pnl_concentration_top_market: float | None = None


@dataclass
class BootstrapResult:
    strategy_id: str
    metric: str
    confidence: float
    low: float | None
    high: float | None
    iterations: int
    sampling_unit: str = "market_id"
