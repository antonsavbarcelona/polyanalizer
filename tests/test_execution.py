from poly_analyzer.execution import round_to_tick, vwap_fill


def test_vwap_single_level():
    vwap, filled = vwap_fill([(0.52, 100.0)], target_size=50.0)
    assert vwap == 0.52
    assert filled == 50.0


def test_vwap_multi_level_matches_manual_calc():
    levels = [(0.52, 20.0), (0.53, 50.0), (0.54, 100.0)]
    vwap, filled = vwap_fill(levels, target_size=100.0)
    expected = (20 * 0.52 + 50 * 0.53 + 30 * 0.54) / 100
    assert abs(vwap - expected) < 1e-9
    assert filled == 100.0


def test_vwap_insufficient_liquidity_partial_fill():
    levels = [(0.52, 20.0), (0.53, 10.0)]
    result = vwap_fill(levels, target_size=100.0)
    assert result is not None
    vwap, filled = result
    assert filled == 30.0
    assert abs(vwap - (20 * 0.52 + 10 * 0.53) / 30) < 1e-9


def test_vwap_empty_book_is_unfilled():
    assert vwap_fill([], target_size=100.0) is None


def test_vwap_zero_size_levels_ignored():
    levels = [(0.50, 0.0), (0.52, 10.0)]
    vwap, filled = vwap_fill(levels, target_size=10.0)
    assert vwap == 0.52
    assert filled == 10.0


def test_vwap_sell_side_uses_bids_descending_for_best_fill():
    """Selling into bids: best (highest) price should fill first."""
    bids_desc = [(0.55, 20.0), (0.54, 50.0), (0.50, 1000.0)]
    vwap, filled = vwap_fill(bids_desc, target_size=70.0)
    expected = (20 * 0.55 + 50 * 0.54) / 70
    assert abs(vwap - expected) < 1e-9
    assert filled == 70.0


def test_round_to_tick():
    assert round_to_tick(0.573, 0.01) == 0.57
    assert round_to_tick(0.576, 0.01) == 0.58
