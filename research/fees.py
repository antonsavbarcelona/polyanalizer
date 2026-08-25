"""Fee model for the offline strategy-discovery analyzer.

IMPORTANT PROVENANCE NOTE: this formula is INFERRED, not taken from any
official Polymarket fee-schedule document (none exists in this repo, and
none was found publicly at the time this module was written). It is
reconstructed from two independent sources that happen to agree closely:

1. Real stored data in `data/analyzer_btc.db`'s `markets` table: the
   `fee_rate` / `fee_exponent` columns observed there are constant across
   all rows at `0.07` / `1.0`. `maker_base_fee` / `taker_base_fee` are ALSO
   observed constant at `1000.0` across every row -- that constancy (no
   variation at all, unlike fee_rate/fee_exponent which are meaningful
   per-market knobs) is why they are treated here as an unused cap/
   placeholder column, not a formula input.
2. `hypo` (repo root), the frozen H1 write-up, states plainly that taker
   fees near price ~0.50 are "~1.75c/share" and maker fees are zero.

Putting `fee_rate=0.07`, `fee_exponent=1.0` into the formula below at
price=0.50 gives exactly 0.0175 (1.75c/share) -- matching hypo's stated
figure exactly, which is why this functional form (a standard
"distance-to-0.5, exponentiated, halved" curve) was chosen over other
candidates that also happen to pass through the single 0.50 data point.

Known discrepancy (left unresolved, not fudged): hypo's fully-worked
example uses entry price 0.531 and states `entry_fee=.0174/share`, but this
formula gives `0.07 * min(0.531, 0.469) ** 1.0 / 2 = 0.016415` at that price
-- about 0.001/share lower than hypo's illustrative number. hypo's numbers
throughout that example are presented as illustrative round figures, not
byte-exact ground truth, so this discrepancy is treated as acceptable and
is pinned (not hidden) in `tests/test_research_fees.py`.

V1 scope: entry is always taker (no maker entry mode exists yet), so only
`entry_fee` uses `taker_fee_per_share`; `exit_fee` branches on
`execution_type` since exits can be MAKER (TP) or TAKER (SL/timeout).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeModel:
    """Stateless; all methods take `market_row` (a dict-like row from the
    `markets` table, or any mapping with `fee_rate`/`fee_exponent` keys) so
    fee behavior can vary per-market without any global config.

    `version` is stored alongside every discovery-pipeline row that used
    this model (contract #12: "FeeModel должен быть versioned") so a future
    change to the fee formula never silently rewrites the meaning of old
    fee_adjusted_response values already on disk."""

    version: str = "fixed_v1"

    def taker_fee_per_share(self, price: float, market_row) -> float:
        fee_rate = market_row["fee_rate"]
        fee_exponent = market_row["fee_exponent"]
        return fee_rate * min(price, 1.0 - price) ** fee_exponent / 2.0

    def maker_fee_per_share(self, price: float, market_row) -> float:
        return 0.0

    def entry_fee(self, price: float, size: float, market_row) -> float:
        # V1: entry is always taker (no maker-entry execution mode exists).
        return self.taker_fee_per_share(price, market_row) * size

    def exit_fee(self, price: float, size: float, execution_type: str, market_row) -> float:
        if execution_type in {"MAKER", "RESOLUTION", "PAYOUT"}:
            return self.maker_fee_per_share(price, market_row) * size
        return self.taker_fee_per_share(price, market_row) * size
