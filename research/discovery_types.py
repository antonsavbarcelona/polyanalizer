"""Canonical dataclasses for the signal-response discovery pipeline.

Deliberately separate from research/types.py (the TP/SL strategy pipeline's
schema): this pipeline is forbidden from ever importing TP/SL/exit-policy
concepts (see research/discovery/README in the package docstring), and a
shared types module would make that boundary easy to violate by accident.

Every "disabled filter" is represented as None, never a sentinel number
(-1, 999999, ...) -- see IMPLEMENTATION CONTRACT #19.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.types import canonical_data, canonical_json, deterministic_id  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# Signal-discovery config (contract #19)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalDiscoveryConfig:
    id: str

    momentum_window_ms: int
    momentum_type: str = "RETURN"  # only "RETURN" implemented in V1 -- see known limitations

    volatility_window_ms: int | None = None
    z_threshold: float | None = None

    absolute_move_threshold_bps: float | None = None

    flow_window_ms: int | None = None
    flow_threshold: float | None = None

    volume_window_ms: int | None = None
    volume_z_threshold: float | None = None

    imbalance_depth: int | None = None
    imbalance_threshold: float | None = None

    poly_lag_window_ms: int | None = None
    max_poly_reprice: float | None = None

    min_contract_price: float | None = None
    max_contract_price: float | None = None

    max_spread: float | None = None

    min_time_remaining_ms: int | None = None
    max_time_remaining_ms: int | None = None

    reset_threshold: float = 1.0
    cooldown_ms: int = 3_000

    def __post_init__(self) -> None:
        if self.momentum_type != "RETURN":
            raise ValueError(
                f"momentum_type={self.momentum_type!r} not implemented in V1 (only 'RETURN' is supported)"
            )
        if (self.volatility_window_ms is None) != (self.z_threshold is None):
            raise ValueError("volatility_window_ms and z_threshold must be both set or both None")
        if (self.flow_window_ms is None) != (self.flow_threshold is None):
            raise ValueError("flow_window_ms and flow_threshold must be both set or both None")
        if (self.volume_window_ms is None) != (self.volume_z_threshold is None):
            raise ValueError("volume_window_ms and volume_z_threshold must be both set or both None")
        if (self.imbalance_depth is None) != (self.imbalance_threshold is None):
            raise ValueError("imbalance_depth and imbalance_threshold must be both set or both None")
        if self.z_threshold is None and self.absolute_move_threshold_bps is None:
            raise ValueError("at least one of z_threshold / absolute_move_threshold_bps must be set")


# ---------------------------------------------------------------------------
# Regimes (contract #28) -- fixed boundaries, [left inclusive, right exclusive)
# except the last bucket in each list, which is right-inclusive/unbounded.
# ---------------------------------------------------------------------------

TTE_REGIME_BOUNDS_S: tuple[tuple[float, float, str], ...] = (
    (0, 30, "0-30s"),
    (30, 60, "30-60s"),
    (60, 120, "60-120s"),
    (120, 300, "120-300s"),
    (300, 600, "300-600s"),
    (600, float("inf"), "600s+"),
)

PRICE_REGIME_BOUNDS: tuple[tuple[float, float, str], ...] = (
    (0.00, 0.20, "0.00-0.20"),
    (0.20, 0.35, "0.20-0.35"),
    (0.35, 0.50, "0.35-0.50"),
    (0.50, 0.65, "0.50-0.65"),
    (0.65, 0.80, "0.65-0.80"),
    (0.80, 1.00, "0.80-1.00"),
)

SPREAD_REGIME_BOUNDS: tuple[tuple[float, float, str], ...] = (
    (0.00, 0.01, "0-1c"),
    (0.01, 0.02, "1-2c"),
    (0.02, 0.03, "2-3c"),
    (0.03, 0.05, "3-5c"),
    (0.05, float("inf"), "5c+"),
)

VOLATILITY_REGIME_LABELS: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "EXTREME")


@dataclass(frozen=True)
class VolatilityRegimeBoundaries:
    """Per-asset quantile boundaries (P25/P75/P95) computed once over the
    DISCOVERY dataset and frozen into experiment metadata -- contract #28
    forbids recomputing them per SignalConfig."""

    asset: str
    p25: float
    p75: float
    p95: float

    def classify(self, value: float) -> str:
        if value <= self.p25:
            return "LOW"
        if value <= self.p75:
            return "MEDIUM"
        if value <= self.p95:
            return "HIGH"
        return "EXTREME"


# ---------------------------------------------------------------------------
# Signal snapshot (contract #20) -- wide feature capture at signal_ts
# ---------------------------------------------------------------------------

@dataclass
class SignalSnapshot:
    signal_id: str
    signal_config_id: str

    asset: str
    market_id: str
    direction: str

    signal_ts: int

    binance_mid: float | None

    return_100ms: float | None
    return_250ms: float | None
    return_500ms: float | None
    return_750ms: float | None
    return_1s: float | None
    return_1_5s: float | None
    return_2s: float | None
    return_3s: float | None
    return_5s: float | None
    return_10s: float | None

    active_momentum_value: float | None

    vol_10s: float | None
    vol_30s: float | None
    vol_60s: float | None
    vol_120s: float | None
    vol_300s: float | None

    active_volatility: float | None

    z_score: float | None

    flow_100ms: float | None
    flow_250ms: float | None
    flow_500ms: float | None
    flow_750ms: float | None
    flow_1s: float | None
    flow_2s: float | None
    flow_3s: float | None
    flow_5s: float | None
    flow_10s: float | None

    active_flow: float | None

    volume_z: float | None

    imbalance_top1: float | None
    imbalance_top3: float | None
    imbalance_top5: float | None
    imbalance_top10: float | None

    microprice_bias: float | None

    target_bid: float | None
    target_ask: float | None
    target_spread: float | None
    target_bid_size: float | None
    target_ask_size: float | None

    poly_move_100ms: float | None
    poly_move_250ms: float | None
    poly_move_500ms: float | None
    poly_move_1s: float | None
    poly_move_2s: float | None

    time_remaining_ms: int | None
    time_elapsed_ms: int | None

    reference_price: float | None
    chainlink_twap: float | None
    distance_to_reference_bps: float | None

    volatility_regime: str | None
    tte_regime: str
    price_regime: str | None
    spread_regime: str | None


# ---------------------------------------------------------------------------
# Entry mark (contract #21)
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryEntryMark:
    entry_mark_id: str
    signal_id: str

    latency_ms: int
    size_shares: float

    entry_target_ts: int
    entry_actual_ts: int | None
    entry_delay_after_target_ms: int | None

    entry_best_bid: float | None
    entry_best_ask: float | None

    entry_vwap: float | None
    entry_slippage: float | None

    available_ask_liquidity: float | None

    entry_fee_total: float | None
    entry_fee_per_share: float | None

    status: str  # EXECUTED | NOT_EXECUTABLE | NO_DATA


# ---------------------------------------------------------------------------
# Signal response (contract #22)
# ---------------------------------------------------------------------------

@dataclass
class SignalResponse:
    response_id: str
    entry_mark_id: str

    signal_id: str
    signal_config_id: str

    asset: str
    market_id: str
    direction: str

    latency_ms: int
    size_shares: float
    horizon_ms: int

    response_target_ts: int
    response_actual_ts: int | None
    response_delay_ms: int | None

    future_best_bid: float | None
    future_best_ask: float | None

    future_sell_vwap: float | None
    available_bid_liquidity: float | None

    raw_response: float | None
    fee_adjusted_response: float | None

    response_positive: bool | None
    fee_adjusted_positive: bool | None

    status: str  # AVAILABLE | AFTER_MARKET_END | NO_DATA | NOT_SELL_EXECUTABLE


# ---------------------------------------------------------------------------
# Path stats (contract #23)
# ---------------------------------------------------------------------------

LEVELS = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.070, 0.100)


def level_field_suffix(level: float) -> str:
    """0.005 -> '005', 0.010 -> '010', 0.100 -> '100' (3-digit cent code)."""
    return f"{round(level * 1000):03d}"


@dataclass
class SignalPathStats:
    entry_mark_id: str
    stats_horizon_ms: int

    mfe: float | None
    mae: float | None

    time_to_mfe_ms: int | None
    time_to_mae_ms: int | None

    time_to_plus_ms: dict[float, int | None] = field(default_factory=dict)  # keyed by LEVELS
    time_to_minus_ms: dict[float, int | None] = field(default_factory=dict)  # keyed by LEVELS


# ---------------------------------------------------------------------------
# Controls (contract #24-27)
# ---------------------------------------------------------------------------

@dataclass
class Control:
    control_id: str
    source_signal_id: str
    signal_config_id: str

    asset: str
    market_id: str
    direction: str

    control_ts: int

    match_rank: int
    match_distance: float

    tte_regime: str
    price_regime: str
    spread_regime: str
    volatility_regime: str | None


@dataclass
class ControlEntryMark:
    control_entry_mark_id: str
    control_id: str

    latency_ms: int
    size_shares: float

    entry_target_ts: int
    entry_actual_ts: int | None
    entry_delay_after_target_ms: int | None

    entry_best_bid: float | None
    entry_best_ask: float | None

    entry_vwap: float | None
    entry_slippage: float | None

    available_ask_liquidity: float | None

    entry_fee_total: float | None
    entry_fee_per_share: float | None

    status: str


@dataclass
class ControlResponse:
    control_response_id: str
    control_entry_mark_id: str

    control_id: str
    signal_config_id: str

    asset: str
    market_id: str
    direction: str

    latency_ms: int
    size_shares: float
    horizon_ms: int

    response_target_ts: int
    response_actual_ts: int | None
    response_delay_ms: int | None

    future_best_bid: float | None
    future_best_ask: float | None

    future_sell_vwap: float | None
    available_bid_liquidity: float | None

    raw_response: float | None
    fee_adjusted_response: float | None

    response_positive: bool | None
    fee_adjusted_positive: bool | None

    status: str


# ---------------------------------------------------------------------------
# Summary tables (contract #29-32)
# ---------------------------------------------------------------------------

# Sentinel `asset` value for the cross-asset pooled summary rows. A plain SQL
# NULL was used originally, but SQL never treats NULL = NULL for uniqueness,
# so INSERT OR REPLACE on the (..., asset, ...) PRIMARY KEY never found the
# "conflicting" prior ALL-row across repeated pooling runs and silently
# duplicated it instead. A real sentinel value fixes that.
ALL_ASSET = "ALL"


@dataclass
class SignalConfigSummary:
    signal_config_id: str
    asset: str | None
    latency_ms: int
    size_shares: float

    signal_count: int
    market_count: int

    executable_entry_count: int
    not_executable_entry_count: int
    entry_execution_rate: float | None

    mean_entry_delay_ms: float | None
    median_entry_delay_ms: float | None
    mean_entry_slippage: float | None
    median_entry_slippage: float | None

    control_count: int

    plateau_score: float | None = None


@dataclass
class SignalResponseSummary:
    signal_config_id: str
    asset: str | None
    latency_ms: int
    size_shares: float
    horizon_ms: int

    available_count: int
    unavailable_count: int
    positive_count: int
    negative_count: int
    zero_count: int
    p_positive: float | None

    mean_response: float | None
    median_response: float | None
    std_response: float | None

    p05: float | None
    p10: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p90: float | None
    p95: float | None

    mean_fee_adjusted_response: float | None
    median_fee_adjusted_response: float | None

    control_available_count: int
    control_p_positive: float | None
    control_mean_response: float | None
    control_median_response: float | None

    uplift_p_positive: float | None
    uplift_mean_response: float | None
    uplift_median_response: float | None

    bootstrap_ci95_mean_low: float | None = None
    bootstrap_ci95_mean_high: float | None = None
    bootstrap_ci95_p_positive_low: float | None = None
    bootstrap_ci95_p_positive_high: float | None = None
    # CI on the UPLIFT itself (signal mean - control mean), not just each
    # side separately -- one of discovery's main deliverables, so it needs
    # its own paired bootstrap rather than being inferred from the two CIs
    # above (which would ignore that signal/control draws are paired by
    # market and can't just be subtracted).
    bootstrap_ci95_uplift_low: float | None = None
    bootstrap_ci95_uplift_high: float | None = None


@dataclass
class SignalHitSummary:
    signal_config_id: str
    asset: str | None
    latency_ms: int
    size_shares: float
    horizon_ms: int
    level: float
    direction: str  # "FAVORABLE" | "ADVERSE"

    sample_count: int
    hit_count: int
    hit_probability: float | None


@dataclass
class SignalFirstPassageSummary:
    signal_config_id: str
    asset: str | None
    latency_ms: int
    size_shares: float
    horizon_ms: int
    plus_level: float
    minus_level: float

    sample_count: int

    # plus_first_count/minus_first_count are the "at least reached, and did
    # so before the other side (or the other side never reached at all)"
    # totals -- i.e. they INCLUDE plus_only_count/minus_only_count
    # respectively. The _only_ counts are broken out separately too
    # (contract feedback: "+3c reached, -2c never" is a different
    # trajectory shape than "both reached, +3c first", even though both
    # count toward plus_first_count).
    plus_first_count: int
    minus_first_count: int
    plus_only_count: int
    minus_only_count: int
    ambiguous_count: int
    neither_count: int

    p_plus_first_given_hit: float | None
    p_minus_first_given_hit: float | None
