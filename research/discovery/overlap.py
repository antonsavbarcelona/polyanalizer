"""Signal overlap between two SignalConfigs (IMPLEMENTATION CONTRACT #38):
reveals when a "top-20" leaderboard is actually the same signal family
wearing different threshold numbers."""
from __future__ import annotations

from research.discovery_types import SignalSnapshot

OVERLAP_WINDOW_MS = 250


def compute_overlap(signals_a: list[SignalSnapshot], signals_b: list[SignalSnapshot]) -> float:
    """|signals in the smaller set with a same-market/same-direction match
    within +/-250ms in the other set| / min(len(A), len(B))."""
    if not signals_a or not signals_b:
        return 0.0

    smaller, larger = (signals_a, signals_b) if len(signals_a) <= len(signals_b) else (signals_b, signals_a)
    by_key: dict[tuple[str, str], list[int]] = {}
    for s in larger:
        by_key.setdefault((s.market_id, s.direction), []).append(s.signal_ts)

    matched = 0
    for s in smaller:
        candidates = by_key.get((s.market_id, s.direction), [])
        if any(abs(s.signal_ts - ts) <= OVERLAP_WINDOW_MS for ts in candidates):
            matched += 1

    return matched / min(len(signals_a), len(signals_b))
