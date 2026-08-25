"""Pure, no-I/O order-book math.

Discovery-stage design: the collector does not simulate trades, fills, fees
or PnL -- it only needs to know "what price would X shares of size have
walked to, right now" so market_state rows carry a realistic sense of
depth, not just top-of-book. Everything about entries/exits/thresholds is
post-hoc analysis over the recorded book, never decided live.
"""
from __future__ import annotations


def round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 8)


def vwap_fill(levels: list[tuple[float, float]], target_size: float) -> tuple[float, float] | None:
    """levels: (price, size) ordered best-first (bids desc for selling,
    asks asc for buying).

    Returns (vwap, filled_size); filled_size < target_size on partial fill.
    Returns None if there is no liquidity at all.
    """
    remaining = target_size
    cost = 0.0
    filled = 0.0
    for price, size in levels:
        if size <= 0:
            continue
        take = min(remaining, size)
        cost += take * price
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled <= 0:
        return None
    return cost / filled, filled
