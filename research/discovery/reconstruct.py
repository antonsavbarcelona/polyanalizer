"""Rebuild dataclass rows from DiscoveryRepository SQL rows.

Needed by the pooling pass (task item 3: per-asset checkpoints): once every
asset for a signal_config is COMPLETE -- possibly across separate runs, if
one asset resumed after a crash -- the pooled (asset=ALL_ASSET) ALL summary
must be computed from EVERY asset's data, not just whatever a single
worker happened to hold in memory. Reading it back from the DB (rather
than keeping it in memory across the whole experiment) is what makes this
resume-safe."""
from __future__ import annotations

import dataclasses
import sqlite3

from research.discovery_types import (
    LEVELS,
    ControlResponse,
    DiscoveryEntryMark,
    SignalPathStats,
    SignalResponse,
    level_field_suffix,
)


def _fields(cls) -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(cls))


def entry_mark_from_row(row: sqlite3.Row) -> DiscoveryEntryMark:
    return DiscoveryEntryMark(**{f: row[f] for f in _fields(DiscoveryEntryMark)})


def _bool_or_none(value) -> bool | None:
    return None if value is None else bool(value)


def signal_response_from_row(row: sqlite3.Row) -> SignalResponse:
    data = {f: row[f] for f in _fields(SignalResponse)}
    data["response_positive"] = _bool_or_none(data["response_positive"])
    data["fee_adjusted_positive"] = _bool_or_none(data["fee_adjusted_positive"])
    return SignalResponse(**data)


def control_response_from_row(row: sqlite3.Row) -> ControlResponse:
    data = {f: row[f] for f in _fields(ControlResponse)}
    data["response_positive"] = _bool_or_none(data["response_positive"])
    data["fee_adjusted_positive"] = _bool_or_none(data["fee_adjusted_positive"])
    return ControlResponse(**data)


def path_stats_from_row(row: sqlite3.Row) -> SignalPathStats:
    time_to_plus = {lvl: row[f"time_to_plus_{level_field_suffix(lvl)}_ms"] for lvl in LEVELS}
    time_to_minus = {lvl: row[f"time_to_minus_{level_field_suffix(lvl)}_ms"] for lvl in LEVELS}
    return SignalPathStats(
        entry_mark_id=row["entry_mark_id"], stats_horizon_ms=row["stats_horizon_ms"],
        mfe=row["mfe"], mae=row["mae"], time_to_mfe_ms=row["time_to_mfe_ms"], time_to_mae_ms=row["time_to_mae_ms"],
        time_to_plus_ms=time_to_plus, time_to_minus_ms=time_to_minus,
    )
