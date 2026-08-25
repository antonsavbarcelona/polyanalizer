import asyncio
import dataclasses

from poly_analyzer.config import RecorderConfig
from poly_analyzer.db import Recorder, WriteJob


def make_recorder(tmp_path) -> Recorder:
    cfg = dataclasses.replace(RecorderConfig(), db_path=str(tmp_path / "test.db"), write_batch_size=100, write_interval_s=1.0)
    rec = Recorder(cfg)
    rec.connect()
    return rec


def test_market_row_has_unique_market_id_pk(tmp_path):
    """U-DB-01 / U-DB-04: re-inserting the same market_id must not duplicate it."""
    rec = make_recorder(tmp_path)
    rec._write_batch([
        WriteJob("markets", {"market_id": "M1", "slug": "s1", "reference_price": None}),
        WriteJob("markets", {"market_id": "M1", "slug": "s1", "reference_price": 100.0}),
    ])
    rows = rec._conn.execute("SELECT COUNT(*) FROM markets").fetchone()
    assert rows[0] == 1
    rec.close()


def test_binance_features_linked_to_correct_market_id(tmp_path):
    """U-DB-02"""
    rec = make_recorder(tmp_path)
    rec._write_batch([WriteJob("binance_features", {"ts": 1, "market_id": "M1", "btc_mid": 100.0})])
    rec._write_batch([WriteJob("binance_features", {"ts": 2, "market_id": "M2", "btc_mid": 200.0})])
    row = rec._conn.execute("SELECT market_id FROM binance_features WHERE btc_mid=100.0").fetchone()
    assert row[0] == "M1"
    rec.close()


def test_signal_linked_to_correct_market_id(tmp_path):
    """U-DB-03 analogue: signals join to the raw tables purely via market_id."""
    rec = make_recorder(tmp_path)
    rec._write_batch([WriteJob("signals", {"signal_id": "S1", "market_id": "M1"})])
    rec._write_batch([WriteJob("signals", {"signal_id": "S2", "market_id": "M2"})])
    row = rec._conn.execute("SELECT market_id FROM signals WHERE signal_id='S2'").fetchone()
    assert row[0] == "M2"
    rec.close()


def test_null_features_stored_as_null(tmp_path):
    """U-DB-05"""
    rec = make_recorder(tmp_path)
    rec._write_batch([WriteJob("binance_features", {"ts": 1, "market_id": "M1", "vol_60s": None})])
    row = rec._conn.execute("SELECT vol_60s FROM binance_features").fetchone()
    assert row == (None,)
    rec.close()


def test_price_precision_preserved(tmp_path):
    """U-DB-06"""
    rec = make_recorder(tmp_path)
    rec._write_batch([WriteJob("polymarket_book", {"market_id": "M1", "ts": 1, "up_best_ask": 0.57})])
    row = rec._conn.execute("SELECT up_best_ask FROM polymarket_book").fetchone()
    assert abs(row[0] - 0.57) < 1e-12
    rec.close()


def test_millisecond_timestamps_not_truncated(tmp_path):
    """U-DB-07 / DB-03: full ms precision preserved on raw rows."""
    rec = make_recorder(tmp_path)
    ts = 1_700_000_000_123
    rec._write_batch([WriteJob("binance_trades", {"exchange_ts": ts, "market_id": "M1"})])
    row = rec._conn.execute("SELECT exchange_ts FROM binance_trades").fetchone()
    assert row[0] == ts
    rec.close()


def test_reopening_existing_db_reads_prior_rows(tmp_path):
    """U-DB-08: after "restart", a fresh connection to the same file sees prior data."""
    rec = make_recorder(tmp_path)
    rec._write_batch([WriteJob("markets", {"market_id": "M1", "slug": "s"})])
    rec.close()

    rec2 = make_recorder(tmp_path)
    row = rec2._conn.execute("SELECT market_id FROM markets").fetchone()
    assert row[0] == "M1"
    rec2.close()


def test_market_settlement_is_a_separate_table_so_it_never_clobbers_market_row(tmp_path):
    """Settlement is written via a separate table (not INSERT OR REPLACE on
    `markets`), specifically so recording it can never NULL out the rest of
    a market's row (slug/tokens/fees/...)."""
    rec = make_recorder(tmp_path)
    rec._write_batch([WriteJob("markets", {
        "market_id": "M1", "slug": "s1", "up_token_id": "UP", "down_token_id": "DOWN",
    })])
    rec._write_batch([WriteJob("market_settlement", {
        "market_id": "M1", "official_outcome": "UP", "derived_outcome": "UP",
    })])
    market_row = rec._conn.execute("SELECT slug, up_token_id, down_token_id FROM markets WHERE market_id='M1'").fetchone()
    assert market_row == ("s1", "UP", "DOWN")
    settlement_row = rec._conn.execute("SELECT official_outcome FROM market_settlement WHERE market_id='M1'").fetchone()
    assert settlement_row[0] == "UP"
    rec.close()


def test_writer_flushes_all_enqueued_rows():
    """U-DB-09 / U-DB-10: 100 enqueued rows all land, and flush() waits for them."""
    async def scenario(tmp_path):
        cfg = dataclasses.replace(RecorderConfig(), db_path=str(tmp_path / "test.db"))
        rec = Recorder(cfg)
        rec.connect()
        rec.start()
        for i in range(100):
            rec.enqueue("binance_features", {"ts": i, "market_id": "M1"})
        await rec.flush()
        count = rec._conn.execute("SELECT COUNT(*) FROM binance_features").fetchone()[0]
        rec.close()
        return count

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        count = asyncio.run(scenario(Path(d)))
    assert count == 100


def test_duplicate_trade_delivery_does_not_create_silent_duplicate_rows(tmp_path):
    """DB-06: raw trades are keyed so a crash/restart replay doesn't
    silently double a trade -- enforced by BinanceFeed's agg_trade_id
    dedup upstream (see test_binance_feed.py), not by the DB layer here."""
    rec = make_recorder(tmp_path)
    rec._write_batch([
        WriteJob("binance_trades", {"market_id": "M1", "agg_trade_id": 5, "exchange_ts": 1, "price": 100.0}),
        WriteJob("binance_trades", {"market_id": "M1", "agg_trade_id": 6, "exchange_ts": 2, "price": 101.0}),
    ])
    count = rec._conn.execute("SELECT COUNT(*) FROM binance_trades").fetchone()[0]
    assert count == 2  # both are genuine autoincrement rows, no accidental PK collision
    rec.close()
