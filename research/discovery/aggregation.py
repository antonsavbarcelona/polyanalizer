"""Summary tables (IMPLEMENTATION CONTRACT #29-33). Pure functions over
already-computed rows -- no DB access here, so these are directly unit
testable against hand-built lists.

Denominator rule (contract #14) applies throughout: every P(R>0)/mean/
median is computed ONLY over responses with status=="AVAILABLE" (i.e.
entry was executable AND a sellable future state existed) -- never over
"all generated signals". availability counts/rates are reported alongside
so bad liquidity can never silently hide inside a rate.

Bootstrap resampling unit is strictly market_id (contract #33): each
iteration samples market_ids with replacement, includes every observation
belonging to the sampled markets, and recomputes the metric -- never
resampling individual signals. A fixed seed makes repeated runs identical.
"""
from __future__ import annotations

import random
from collections import defaultdict
from statistics import median
from typing import Iterable

from research.discovery_types import (
    Control,
    ControlResponse,
    DiscoveryEntryMark,
    SignalConfigSummary,
    SignalFirstPassageSummary,
    SignalHitSummary,
    SignalPathStats,
    SignalResponse,
    SignalResponseSummary,
)


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stdev(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    return (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5


# ---------------------------------------------------------------------------
# signal_config_summary (contract #29)
# ---------------------------------------------------------------------------

def compute_signal_config_summary(
    signal_config_id: str, asset: str | None, latency_ms: int, size_shares: float,
    entry_marks: Iterable[DiscoveryEntryMark], market_ids: set[str], control_count: int,
) -> SignalConfigSummary:
    marks = list(entry_marks)
    signal_count = len(marks)
    executable = [m for m in marks if m.status == "EXECUTED"]
    executable_entry_count = len(executable)
    slippages = [m.entry_slippage for m in executable if m.entry_slippage is not None]
    delays = [m.entry_delay_after_target_ms for m in executable if m.entry_delay_after_target_ms is not None]

    return SignalConfigSummary(
        signal_config_id=signal_config_id, asset=asset, latency_ms=latency_ms, size_shares=size_shares,
        signal_count=signal_count, market_count=len(market_ids),
        executable_entry_count=executable_entry_count,
        not_executable_entry_count=signal_count - executable_entry_count,
        entry_execution_rate=(executable_entry_count / signal_count) if signal_count else None,
        mean_entry_delay_ms=_mean(delays),
        median_entry_delay_ms=median(delays) if delays else None,
        mean_entry_slippage=_mean(slippages),
        median_entry_slippage=median(slippages) if slippages else None,
        control_count=control_count,
    )


# ---------------------------------------------------------------------------
# signal_response_summary (contract #30) + bootstrap (contract #33)
# ---------------------------------------------------------------------------

def _response_stats(available: list[SignalResponse | ControlResponse]) -> dict:
    raw = sorted(r.raw_response for r in available)
    fee_adj = [r.fee_adjusted_response for r in available if r.fee_adjusted_response is not None]
    positive = sum(1 for r in available if r.response_positive)
    n = len(available)
    return dict(
        available_count=n,
        positive_count=positive,
        negative_count=sum(1 for r in available if r.raw_response < 0),
        zero_count=sum(1 for r in available if r.raw_response == 0),
        p_positive=(positive / n) if n else None,
        mean_response=_mean(raw),
        median_response=median(raw) if raw else None,
        std_response=_stdev(raw),
        p05=_percentile(raw, 0.05), p10=_percentile(raw, 0.10), p25=_percentile(raw, 0.25),
        p50=_percentile(raw, 0.50), p75=_percentile(raw, 0.75), p90=_percentile(raw, 0.90),
        p95=_percentile(raw, 0.95),
        mean_fee_adjusted_response=_mean(fee_adj),
        median_fee_adjusted_response=median(fee_adj) if fee_adj else None,
    )


def _bootstrap_ci(
    available: list[SignalResponse], market_id_of: dict, control_available: list[ControlResponse],
    control_market_id_of: dict, *, iterations: int, seed: int, confidence: float = 0.95,
) -> tuple[tuple[float | None, float | None], tuple[float | None, float | None], tuple[float | None, float | None]]:
    """((mean_low, mean_high), (p_positive_low, p_positive_high), (uplift_low, uplift_high)).

    The uplift CI resamples market_ids ONCE per iteration and applies that
    SAME draw to both the signal and control groups (not two independent
    resamples) -- signal and control observations are paired by market, and
    subtracting two independently-resampled means would overstate the
    uplift's true variance."""
    groups: dict[str, list[SignalResponse]] = defaultdict(list)
    for r in available:
        groups[market_id_of[id(r)]].append(r)
    control_groups: dict[str, list[ControlResponse]] = defaultdict(list)
    for r in control_available:
        control_groups[control_market_id_of[id(r)]].append(r)

    market_ids = sorted(set(groups) | set(control_groups))
    if not market_ids or iterations <= 0:
        return (None, None), (None, None), (None, None)

    rng = random.Random(seed)
    mean_samples: list[float] = []
    p_pos_samples: list[float] = []
    uplift_samples: list[float] = []
    for _ in range(iterations):
        drawn = [rng.choice(market_ids) for _ in market_ids]
        sampled: list[SignalResponse] = []
        control_sampled: list[ControlResponse] = []
        for market_id in drawn:
            sampled.extend(groups.get(market_id, []))
            control_sampled.extend(control_groups.get(market_id, []))

        if sampled:
            mean_samples.append(sum(r.raw_response for r in sampled) / len(sampled))
            p_pos_samples.append(sum(1 for r in sampled if r.response_positive) / len(sampled))
        if sampled and control_sampled:
            sample_mean = sum(r.raw_response for r in sampled) / len(sampled)
            control_mean = sum(r.raw_response for r in control_sampled) / len(control_sampled)
            uplift_samples.append(sample_mean - control_mean)

    mean_samples.sort()
    p_pos_samples.sort()
    uplift_samples.sort()
    alpha = (1.0 - confidence) / 2.0
    mean_ci = (_percentile(mean_samples, alpha), _percentile(mean_samples, 1.0 - alpha))
    p_pos_ci = (_percentile(p_pos_samples, alpha), _percentile(p_pos_samples, 1.0 - alpha))
    uplift_ci = (_percentile(uplift_samples, alpha), _percentile(uplift_samples, 1.0 - alpha))
    return mean_ci, p_pos_ci, uplift_ci


def compute_signal_response_summary(
    signal_config_id: str, asset: str | None, latency_ms: int, size_shares: float, horizon_ms: int,
    responses: list[SignalResponse], response_market_id: dict, control_responses: list[ControlResponse],
    control_response_market_id: dict,
    *, bootstrap_iterations: int = 1_000, bootstrap_seed: int = 1337,
) -> SignalResponseSummary:
    available = [r for r in responses if r.status == "AVAILABLE" and r.raw_response is not None]
    stats = _response_stats(available)

    control_available = [r for r in control_responses if r.status == "AVAILABLE" and r.raw_response is not None]
    control_stats = _response_stats(control_available)

    uplift_p_positive = (
        stats["p_positive"] - control_stats["p_positive"]
        if stats["p_positive"] is not None and control_stats["p_positive"] is not None else None
    )
    uplift_mean_response = (
        stats["mean_response"] - control_stats["mean_response"]
        if stats["mean_response"] is not None and control_stats["mean_response"] is not None else None
    )
    uplift_median_response = (
        stats["median_response"] - control_stats["median_response"]
        if stats["median_response"] is not None and control_stats["median_response"] is not None else None
    )

    market_id_of = {id(r): response_market_id[id(r)] for r in available}
    control_market_id_of = {id(r): control_response_market_id[id(r)] for r in control_available}
    mean_ci, p_pos_ci, uplift_ci = _bootstrap_ci(
        available, market_id_of, control_available, control_market_id_of,
        iterations=bootstrap_iterations, seed=bootstrap_seed,
    )

    return SignalResponseSummary(
        signal_config_id=signal_config_id, asset=asset, latency_ms=latency_ms, size_shares=size_shares,
        horizon_ms=horizon_ms,
        available_count=stats["available_count"], unavailable_count=len(responses) - stats["available_count"],
        positive_count=stats["positive_count"],
        negative_count=stats["negative_count"], zero_count=stats["zero_count"], p_positive=stats["p_positive"],
        mean_response=stats["mean_response"], median_response=stats["median_response"],
        std_response=stats["std_response"],
        p05=stats["p05"], p10=stats["p10"], p25=stats["p25"], p50=stats["p50"], p75=stats["p75"],
        p90=stats["p90"], p95=stats["p95"],
        mean_fee_adjusted_response=stats["mean_fee_adjusted_response"],
        median_fee_adjusted_response=stats["median_fee_adjusted_response"],
        control_available_count=control_stats["available_count"], control_p_positive=control_stats["p_positive"],
        control_mean_response=control_stats["mean_response"], control_median_response=control_stats["median_response"],
        uplift_p_positive=uplift_p_positive, uplift_mean_response=uplift_mean_response,
        uplift_median_response=uplift_median_response,
        bootstrap_ci95_mean_low=mean_ci[0], bootstrap_ci95_mean_high=mean_ci[1],
        bootstrap_ci95_p_positive_low=p_pos_ci[0], bootstrap_ci95_p_positive_high=p_pos_ci[1],
        bootstrap_ci95_uplift_low=uplift_ci[0], bootstrap_ci95_uplift_high=uplift_ci[1],
    )


# ---------------------------------------------------------------------------
# signal_hit_summary (contract #31)
# ---------------------------------------------------------------------------

def compute_hit_summary(
    signal_config_id: str, asset: str | None, latency_ms: int, size_shares: float, horizon_ms: int,
    level: float, direction: str, path_stats: list[SignalPathStats],
) -> SignalHitSummary:
    assert direction in ("FAVORABLE", "ADVERSE")
    sample_count = len(path_stats)
    if direction == "FAVORABLE":
        hit_count = sum(1 for s in path_stats if s.time_to_plus_ms.get(level) is not None)
    else:
        hit_count = sum(1 for s in path_stats if s.time_to_minus_ms.get(level) is not None)

    return SignalHitSummary(
        signal_config_id=signal_config_id, asset=asset, latency_ms=latency_ms, size_shares=size_shares,
        horizon_ms=horizon_ms, level=level, direction=direction,
        sample_count=sample_count, hit_count=hit_count,
        hit_probability=(hit_count / sample_count) if sample_count else None,
    )


# ---------------------------------------------------------------------------
# signal_first_passage_summary (contract #17, #32)
# ---------------------------------------------------------------------------

def _first_passage_outcome(stats: SignalPathStats, plus_level: float, minus_level: float) -> str:
    t_plus = stats.time_to_plus_ms.get(plus_level)
    t_minus = stats.time_to_minus_ms.get(minus_level)
    if t_plus is None and t_minus is None:
        return "NEITHER"
    if t_plus is not None and t_minus is None:
        return "PLUS_ONLY"
    if t_minus is not None and t_plus is None:
        return "MINUS_ONLY"
    if t_plus < t_minus:
        return "PLUS_FIRST"
    if t_minus < t_plus:
        return "MINUS_FIRST"
    return "AMBIGUOUS"


def compute_first_passage_summary(
    signal_config_id: str, asset: str | None, latency_ms: int, size_shares: float, horizon_ms: int,
    plus_level: float, minus_level: float, path_stats: list[SignalPathStats],
) -> SignalFirstPassageSummary:
    outcomes = [_first_passage_outcome(s, plus_level, minus_level) for s in path_stats]
    plus_only = sum(1 for o in outcomes if o == "PLUS_ONLY")
    minus_only = sum(1 for o in outcomes if o == "MINUS_ONLY")
    plus_first = sum(1 for o in outcomes if o == "PLUS_FIRST") + plus_only
    minus_first = sum(1 for o in outcomes if o == "MINUS_FIRST") + minus_only
    ambiguous = sum(1 for o in outcomes if o == "AMBIGUOUS")
    neither = sum(1 for o in outcomes if o == "NEITHER")

    hit_denominator = plus_first + minus_first + ambiguous  # "at least one barrier reached"
    return SignalFirstPassageSummary(
        signal_config_id=signal_config_id, asset=asset, latency_ms=latency_ms, size_shares=size_shares,
        horizon_ms=horizon_ms, plus_level=plus_level, minus_level=minus_level,
        sample_count=len(path_stats),
        plus_first_count=plus_first, minus_first_count=minus_first,
        plus_only_count=plus_only, minus_only_count=minus_only,
        ambiguous_count=ambiguous, neither_count=neither,
        p_plus_first_given_hit=(plus_first / hit_denominator) if hit_denominator else None,
        p_minus_first_given_hit=(minus_first / hit_denominator) if hit_denominator else None,
    )
