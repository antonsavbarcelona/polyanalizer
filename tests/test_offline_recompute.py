"""OFFLINE-01/02/03/04: the acceptance test for the whole raw-storage
design. Live-computed features and features recomputed offline from
nothing but the raw tables must match exactly -- proving raw storage is
genuinely sufficient for research, per the project's stated main
acceptance criterion (test-cases section P).
"""
import sqlite3
import time

from poly_analyzer.discovery import DiscoverySignalConfig, iter_replay
from poly_analyzer.features import book_imbalance, trade_flow
from poly_analyzer.timeseries import time_return
from tests.discovery_fixtures import build_binance_only_db

MARKET_ID = "M1"

# Depth/book messages carry no exchange timestamp, so BinanceFeed stamps
# them with real wall-clock recv time -- trade exchange_ts must be on the
# same real-time scale (not small fixed integers) or the two series never
# overlap when compared.
def _events(base_ts: int):
    return [
        ("depth", [(100.0, 5.0)], [(100.2, 5.0)]),
        ("trade", 1, base_ts + 1_000, 100.1, 1.0, False),   # aggressive buy
        ("trade", 2, base_ts + 1_200, 100.1, 0.5, False),   # aggressive buy
        ("trade", 3, base_ts + 1_400, 100.0, 2.0, True),    # aggressive sell
        ("depth", [(100.05, 40.0), (100.0, 20.0)], [(100.15, 10.0), (100.2, 5.0)]),
        ("trade", 4, base_ts + 2_500, 100.3, 1.0, False),
        ("depth", [(100.2, 30.0)], [(100.35, 8.0)]),
    ]


def _run(tmp_path):
    """Builds the live state + persisted DB once, replays it offline once,
    and returns (live_state, offline_state) with connections still open --
    caller is responsible for closing both."""
    base_ts = int(time.time() * 1000)
    live_state, rec = build_binance_only_db(tmp_path, MARKET_ID, _events(base_ts))

    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.row_factory = sqlite3.Row
    offline_state = None
    for _now, state, _kind in iter_replay(conn, MARKET_ID, DiscoverySignalConfig(flow_window_ms=2_000)):
        offline_state = state

    return live_state, offline_state, rec, conn, base_ts


def test_offline_01_02_03_04_live_and_offline_features_match(tmp_path):
    live_state, offline_state, rec, conn, base_ts = _run(tmp_path)
    try:
        now_ms = base_ts + 3_000

        # OFFLINE-01: return
        live_return = time_return(live_state.price_history, now_ms, 1_000)
        offline_return = time_return(offline_state.price_history, now_ms, 1_000)
        assert live_return is not None
        assert live_return == offline_return

        # OFFLINE-02: trade flow
        live_flow = trade_flow(live_state, now_ms, 2_000)
        offline_flow = trade_flow(offline_state, now_ms, 2_000)
        assert live_flow is not None
        assert live_flow == offline_flow

        # OFFLINE-03: book imbalance
        live_imbalance = book_imbalance(live_state, 5)
        offline_imbalance = book_imbalance(offline_state, 5)
        assert live_imbalance is not None
        assert live_imbalance == offline_imbalance

        # OFFLINE-04: a feature never computed live at all, e.g. return_750ms,
        # is still fully reconstructable purely from raw storage.
        r750 = time_return(offline_state.price_history, now_ms, 750)
        assert r750 is not None
    finally:
        rec.close()
        conn.close()
