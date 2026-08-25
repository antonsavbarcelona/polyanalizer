"""Fair-value V0: a diffusion baseline, not a real probability model.

hypo.md is explicit that this must NOT gate signal emission -- it exists so
we can later check, without re-running the experiment, whether it would
have added predictive value on top of the deterministic lead-lag signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .timeseries import normal_cdf

CHAINLINK_STALE_MS = 5_000


@dataclass
class FairValue:
    z: float | None
    fair_up: float | None
    fair_down: float | None


def fair_value_v0(reference_price: float | None, current_price: float | None,
                   sigma_1s: float | None, remaining_s: float | None,
                   chainlink_observation_ts_ms: int | None, now_ms: int) -> FairValue:
    if reference_price is None or current_price is None or reference_price <= 0:
        return FairValue(None, None, None)
    if sigma_1s is None or sigma_1s <= 0:
        return FairValue(None, None, None)
    if remaining_s is None:
        return FairValue(None, None, None)
    if chainlink_observation_ts_ms is not None and now_ms - chainlink_observation_ts_ms > CHAINLINK_STALE_MS:
        return FairValue(None, None, None)

    t_remaining = max(remaining_s, 1.0)
    sigma_remaining = sigma_1s * math.sqrt(t_remaining)
    if sigma_remaining <= 0:
        return FairValue(None, None, None)

    z = math.log(current_price / reference_price) / sigma_remaining
    fair_up = normal_cdf(z)
    fair_up = min(max(fair_up, 0.0), 1.0)
    return FairValue(z=z, fair_up=fair_up, fair_down=1.0 - fair_up)
