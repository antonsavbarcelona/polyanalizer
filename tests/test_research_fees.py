"""Pins for research.fees.FeeModel -- see the module docstring in
research/fees.py for the full provenance discussion (inferred formula, not
an official Polymarket doc)."""
import pytest

from research.fees import FeeModel

MARKET = {"fee_rate": 0.07, "fee_exponent": 1.0}


def test_taker_fee_per_share_at_50c_matches_hypo_stated_figure():
    fm = FeeModel()
    # hypo states "~1.75c/share taker fee at price ~0.50" -- exact match here.
    assert fm.taker_fee_per_share(0.50, MARKET) == 0.0175


def test_maker_fee_per_share_is_always_zero():
    fm = FeeModel()
    assert fm.maker_fee_per_share(0.50, MARKET) == 0.0
    assert fm.maker_fee_per_share(0.20, MARKET) == 0.0
    assert fm.maker_fee_per_share(0.80, MARKET) == 0.0


def test_taker_fee_per_share_at_hypo_worked_example_price_is_close_but_not_exact():
    """hypo's fully-worked example (entry_vwap=.531) states entry_fee=.0174/
    share. The actual formula gives a slightly different number -- this is a
    known, documented, and accepted discrepancy (see research/fees.py
    docstring), not a bug to be fudged away."""
    fm = FeeModel()
    actual = fm.taker_fee_per_share(0.531, MARKET)
    assert actual == pytest.approx(0.016415, abs=1e-6)
    # Confirm it is indeed NOT hypo's stated figure (documents the gap).
    assert actual != pytest.approx(0.0174, abs=1e-6)
    # But it IS close to hypo's figure at a loose (order-of-magnitude-correct) tolerance.
    assert actual == pytest.approx(0.0174, abs=2e-3)


def test_entry_fee_scales_by_size_and_is_always_taker():
    fm = FeeModel()
    assert fm.entry_fee(0.50, 100, MARKET) == pytest.approx(1.75)


def test_exit_fee_maker_is_zero_taker_uses_formula():
    fm = FeeModel()
    assert fm.exit_fee(0.55, 100, "MAKER", MARKET) == 0.0
    assert fm.exit_fee(0.50, 100, "TAKER", MARKET) == pytest.approx(1.75)
