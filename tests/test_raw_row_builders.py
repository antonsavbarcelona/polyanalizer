from poly_analyzer.binance_feed import build_binance_book_row
from poly_analyzer.config import MarketConfig
from poly_analyzer.features import build_binance_features_row
from poly_analyzer.polymarket_feed import PolymarketFeed, build_polymarket_book_row
from poly_analyzer.state import MarketState

NOW = 1_700_000_000_000


def test_binance_book_row_has_top10_both_sides_best_first():
    """BIN-BOOK-01/02/03: bids desc, asks asc, top-10 preserved."""
    bids = [(100.0, 1.0), (99.0, 2.0), (98.0, 3.0)]
    asks = [(101.0, 1.5), (102.0, 2.5)]
    row = build_binance_book_row(bids, asks, "M1", None, NOW, 123, is_debug=False)
    assert row["best_bid_price"] == 100.0
    assert row["best_ask_price"] == 101.0
    assert row["bid_1_price"] == 100.0 and row["bid_2_price"] == 99.0 and row["bid_3_price"] == 98.0
    assert row["ask_1_price"] == 101.0 and row["ask_2_price"] == 102.0
    # beyond available depth -> explicit None, not silently wrong/absent
    assert row["bid_4_price"] is None
    assert row["ask_3_price"] is None
    assert row["recv_monotonic_ns"] == 123


def test_binance_book_row_missing_side_is_explicit_none():
    row = build_binance_book_row([], [], "M1", None, NOW, 1, is_debug=False)
    assert row["best_bid_price"] is None and row["best_ask_price"] is None


def test_binance_features_row_covers_the_full_spec_field_set():
    state = MarketState()
    state.price_history.append(NOW - 5_000, 100.0)
    state.price_history.append(NOW, 101.0)
    state.trade_buffer.append((NOW, 10.0, True))
    state.binance_bids = [(99.0, 5.0)]
    state.binance_asks = [(101.0, 5.0)]

    row = build_binance_features_row(state, NOW, "M1", is_debug=True)
    for key in (
        "return_100ms", "return_250ms", "return_500ms", "return_1s", "return_2s", "return_5s",
        "vol_10s", "vol_30s", "vol_60s", "vol_120s",
        "flow_250ms", "flow_500ms", "flow_1s", "flow_2s", "flow_5s",
        "imbalance_top1", "imbalance_top5", "imbalance_top10",
    ):
        assert key in row
    assert row["is_debug"] == 1
    assert row["return_5s"] is not None  # the 5s-old sample makes this one computable


def test_polymarket_book_row_has_both_sides_top10():
    state = MarketState()
    state.market_id = "M1"
    poly = PolymarketFeed(MarketConfig(), state)
    poly.set_tokens("UP", "DOWN")
    poly._book_snapshot["UP"] = ([(0.49, 10.0)], [(0.51, 20.0), (0.52, 5.0)])
    poly._book_snapshot["DOWN"] = ([(0.47, 3.0)], [(0.53, 8.0)])
    state.up.bid, state.up.ask = 0.49, 0.51
    state.down.bid, state.down.ask = 0.47, 0.53

    row = build_polymarket_book_row(state, poly, NOW, 555, is_debug=False)
    assert row["up_best_bid"] == 0.49 and row["up_best_ask"] == 0.51
    assert row["up_ask_1_price"] == 0.51 and row["up_ask_2_price"] == 0.52
    assert row["down_best_bid"] == 0.47 and row["down_ask_1_price"] == 0.53
    assert row["recv_monotonic_ns"] == 555
