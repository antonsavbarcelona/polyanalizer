from poly_analyzer.discovery import ResponseStats, aggregate_responses


def make_stats(final_return, plus_hits=None):
    return ResponseStats(
        entry_ts=0, entry_vwap=0.5, size=100,
        mfe=final_return, mae=final_return,
        time_to_mfe_ms=None, time_to_mae_ms=None,
        time_to_plus_ms=plus_hits or {0.01: None},
        final_return=final_return,
    )


def test_aggregate_basic_stats():
    responses = [make_stats(0.02), make_stats(-0.01), make_stats(0.03), make_stats(0.0)]
    agg = aggregate_responses(responses, market_ids={"M1", "M2"})
    assert agg.signal_count == 4
    assert agg.market_count == 2
    assert agg.signals_per_market == 2.0
    assert abs(agg.mean_return - (0.02 - 0.01 + 0.03 + 0.0) / 4) < 1e-9
    assert agg.p_positive == 0.5  # two of four are > 0


def test_aggregate_p_reach_counts_signals_that_touched_the_level():
    responses = [
        make_stats(0.02, plus_hits={0.01: 100}),
        make_stats(-0.01, plus_hits={0.01: None}),
        make_stats(0.015, plus_hits={0.01: 250}),
    ]
    agg = aggregate_responses(responses, market_ids={"M1"})
    assert abs(agg.p_reach[0.01] - 2 / 3) < 1e-9


def test_aggregate_empty_is_safe():
    agg = aggregate_responses([], market_ids=set())
    assert agg.signal_count == 0
    assert agg.mean_return is None
    assert agg.p_positive is None
