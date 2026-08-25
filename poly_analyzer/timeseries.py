"""Time-indexed rolling series.

All lookups are by timestamp, never by tick count -- ticks arrive at
irregular intervals (Binance aggTrade only fires on trades, book updates
are event-driven), so "N samples ago" is not "N seconds ago".
"""
from __future__ import annotations

import bisect
import math


class TimeSeries:
    """(ts_ms, value) samples kept for at most `max_age_ms`.

    Backed by two parallel lists (not a deque of tuples) so `value_before`
    can binary-search instead of linearly scanning: during offline signal
    replay (discovery.py's resampled_returns) this lookup runs millions of
    times per (signal_config, market) pair, and profiling showed the old
    O(n) linear scan accounting for >90% of total replay wall time.
    Timestamps are non-decreasing by construction (out-of-order appends are
    rejected below), so bisect is always valid.

    Expiry is lazy: `_start` marks the first live index, and old entries
    are only physically dropped (`del ...[:_start]`) once they're a
    sizeable fraction of the list, so trimming stays amortized O(1) instead
    of shifting the whole list on every append.
    """

    def __init__(self, max_age_ms: int):
        self.max_age_ms = max_age_ms
        self._ts: list[int] = []
        self._values: list[float] = []
        self._start = 0

    def append(self, ts_ms: int, value: float) -> None:
        if self._ts and ts_ms < self._ts[-1]:
            # Out-of-order arrival: ignore rather than corrupt ordering.
            return
        self._ts.append(ts_ms)
        self._values.append(value)
        cutoff = ts_ms - self.max_age_ms
        start = self._start
        ts = self._ts
        n = len(ts)
        while start < n and ts[start] < cutoff:
            start += 1
        self._start = start
        if self._start > 1024 and self._start * 2 > len(ts):
            del self._ts[: self._start]
            del self._values[: self._start]
            self._start = 0

    def __len__(self) -> int:
        return len(self._ts) - self._start

    @property
    def latest(self) -> tuple[int, float] | None:
        if len(self._ts) <= self._start:
            return None
        return self._ts[-1], self._values[-1]

    def value_before(self, ts_ms: int) -> tuple[int, float] | None:
        """Most recent sample with sample_ts <= ts_ms, or None if none exists."""
        idx = bisect.bisect_right(self._ts, ts_ms, self._start) - 1
        if idx < self._start:
            return None
        return self._ts[idx], self._values[idx]

    def values_in_window(self, start_ts_ms: int, end_ts_ms: int) -> list[tuple[int, float]]:
        lo = bisect.bisect_left(self._ts, start_ts_ms, self._start)
        hi = bisect.bisect_right(self._ts, end_ts_ms, lo)
        return list(zip(self._ts[lo:hi], self._values[lo:hi]))


def time_return(series: TimeSeries, now_ts_ms: int, lookback_ms: int) -> float | None:
    """(now - past) / past using the most recent sample at/after `lookback_ms` ago.

    Returns None ("not ready") if there is no sample old enough, matching
    U-FEAT-04: insufficient history must not silently produce 0.
    """
    latest = series.value_before(now_ts_ms)
    if latest is None:
        return None
    past = series.value_before(now_ts_ms - lookback_ms)
    if past is None:
        return None
    past_ts, past_val = past
    if past_val == 0:
        return None
    _, now_val = latest
    return (now_val - past_val) / past_val


def stdev(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
