import dataclasses
import sqlite3

from poly_analyzer.config import CONFIG, RecorderConfig
from poly_analyzer.db import Recorder, WriteJob
from poly_analyzer.settlement import compute_derived_outcome, parse_official_outcome


def test_parses_up_win_from_gamma_style_encoded_strings():
    """Gamma API double-encodes outcomes/outcomePrices as JSON strings,
    exactly like clobTokenIds -- confirmed live against a resolved market."""
    raw = {
        "umaResolutionStatus": "resolved",
        "outcomes": '["Up", "Down"]',
        "outcomePrices": '["1", "0"]',
    }
    assert parse_official_outcome(raw) == "UP"


def test_parses_down_win():
    raw = {"umaResolutionStatus": "resolved", "outcomes": '["Up", "Down"]', "outcomePrices": '["0", "1"]'}
    assert parse_official_outcome(raw) == "DOWN"


def test_unresolved_market_returns_none():
    raw = {"umaResolutionStatus": "pending", "outcomes": '["Up", "Down"]', "outcomePrices": '["0.5", "0.5"]'}
    assert parse_official_outcome(raw) is None


def test_accepts_already_decoded_lists_too():
    raw = {"umaResolutionStatus": "resolved", "outcomes": ["Up", "Down"], "outcomePrices": ["1", "0"]}
    assert parse_official_outcome(raw) == "UP"


def test_derived_outcome_none_without_reference_price(tmp_path):
    outcome, twap = compute_derived_outcome(str(tmp_path / "x.db"), "M1", 1000, reference_price=None)
    assert outcome is None and twap is None


def _make_db_with_observations(tmp_path, rows):
    cfg = dataclasses.replace(RecorderConfig(), db_path=str(tmp_path / "test.db"))
    rec = Recorder(cfg)
    rec.connect()
    rec._write_batch([
        WriteJob("chainlink_observations", {"observation_ts": r["ts"], "market_id": r["market_id"],
                                             "twap_value": r["chainlink_twap"]})
        for r in rows
    ])
    rec.close()
    return cfg.db_path


def test_derived_outcome_uses_first_observation_at_or_after_end_ts_not_before(tmp_path):
    """No-lookahead for settlement math: must use the first TWAP sample
    AT-OR-AFTER market end, never a later-but-closer-in-time-error sample
    from before it (that would be look-ahead in reverse -- ignoring data
    that actually existed at settlement time in favor of stale data)."""
    end_ts = 2_000
    db_path = _make_db_with_observations(tmp_path, [
        {"ts": 1_800, "market_id": "M1", "chainlink_twap": 90.0},   # before end: must be ignored
        {"ts": 2_050, "market_id": "M1", "chainlink_twap": 105.0},  # first sample at/after end
        {"ts": 2_300, "market_id": "M1", "chainlink_twap": 110.0},  # later: must be ignored
    ])
    outcome, twap = compute_derived_outcome(db_path, "M1", end_ts, reference_price=100.0)
    assert twap == 105.0
    assert outcome == "UP"


def test_derived_outcome_down_when_twap_below_reference(tmp_path):
    db_path = _make_db_with_observations(tmp_path, [
        {"ts": 2_050, "market_id": "M1", "chainlink_twap": 95.0},
    ])
    outcome, twap = compute_derived_outcome(db_path, "M1", 2_000, reference_price=100.0)
    assert outcome == "DOWN"


def test_derived_outcome_falls_back_to_last_known_sample_if_none_after_end(tmp_path):
    db_path = _make_db_with_observations(tmp_path, [
        {"ts": 1_500, "market_id": "M1", "chainlink_twap": 101.0},
        {"ts": 1_900, "market_id": "M1", "chainlink_twap": 103.0},
    ])
    outcome, twap = compute_derived_outcome(db_path, "M1", 2_000, reference_price=100.0)
    assert twap == 103.0
    assert outcome == "UP"
