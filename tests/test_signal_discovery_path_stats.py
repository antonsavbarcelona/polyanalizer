"""Executable-VWAP MFE/MAE + time-to-level (IMPLEMENTATION CONTRACT #15-16),
hand-computed against the same golden fixture used by
test_signal_discovery_golden.py's Signal A."""
from __future__ import annotations

from tests.test_signal_discovery_golden import MARKET_ROW, _build_golden_db, _snapshot

from research.discovery.entry import compute_entry_mark
from research.discovery.path_stats import compute_path_stats
from research.fees import FeeModel


def test_path_stats_mfe_mae_and_time_to_level(tmp_path):
    conn = _build_golden_db(tmp_path)
    signal = _snapshot("SIG_A", "M_A", "UP", signal_ts=1_000)
    entry = compute_entry_mark(conn, signal, latency_ms=250, size_shares=100, market_row=MARKET_ROW,
                                fee_model=FeeModel())
    assert entry.entry_actual_ts == 1_300
    assert abs(entry.entry_vwap - 0.516) < 1e-9

    # Window [1300, 1300+1500=2800]: two book rows fall inside it --
    # the entry row itself (ts=1300, up bid 0.49 x100 -> sell_vwap=0.49,
    # r=-0.026) and ts=2300 (sell_vwap=0.545, r=+0.029, computed in the
    # golden response test).
    stats = compute_path_stats(conn, "M_A", "UP", entry, stats_horizon_ms=1_500, market_row=MARKET_ROW)

    assert abs(stats.mfe - 0.029) < 1e-9
    assert stats.time_to_mfe_ms == 1_000  # reached at ts=2300, entry at ts=1300
    assert abs(stats.mae - (-0.026)) < 1e-9
    assert stats.time_to_mae_ms == 0  # the entry point itself is the worst point in this short window

    for lvl in (0.005, 0.010, 0.015, 0.020, 0.025):
        assert stats.time_to_plus_ms[lvl] == 1_000, lvl
    assert stats.time_to_plus_ms[0.030] is None  # MFE=0.029 never reaches +0.030

    for lvl in (0.005, 0.010, 0.015, 0.020, 0.025):
        assert stats.time_to_minus_ms[lvl] == 0, lvl
    assert stats.time_to_minus_ms[0.030] is None  # MAE=-0.026 never reaches -0.030
